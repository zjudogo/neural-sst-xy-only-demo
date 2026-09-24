// Native OMPL executable for XY-only Neural-SST.
//
// The Python process predicts a P_xy probability map once at the beginning and writes
// it to a task file. This program reads that snapshot and performs all SST expansions,
// vehicle-state propagation, collision checking, and result serialization in C++.
// This allows GPU-accelerated inference while avoiding cross-language calls in the
// planning hot loop. "XY-only" means the network guides only (x, y) sample locations;
// yaw is still sampled uniformly, and goal attainment checks only positional error.

#include <ompl/base/goals/GoalRegion.h>
#include <ompl/base/objectives/ControlDurationObjective.h>
#include <ompl/control/PlannerData.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/util/RandomNumbers.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ob = ompl::base;
namespace oc = ompl::control;

// These geometry constants must exactly match Python's geometry.py: 224 * 0.05 m
// spans an 11.2 m square.
constexpr int kGridSize = 224;
constexpr double kResolutionM = 0.05;
constexpr double kMinM = -5.6;
constexpr double kMaxM = 5.6;
// Fixed forward speed and wheelbase for the kinematic bicycle model; the only
// control input is the front-wheel steering angle.
constexpr double kSpeedMps = 2.0;
constexpr double kWheelbaseM = 0.33;
// OMPL discretizes each control segment into propagations of this duration. The small
// step also makes collision checking more conservative.
constexpr double kPropagationStepS = 0.02;
constexpr double kPi = 3.14159265358979323846;

// Task snapshot matching the write order in Python's run_native. Both occupancy and
// probability are row-major flat arrays, indexed as row(y) * kGridSize + column(x).
struct Input {
    unsigned seed = 0;
    double xy_bias = 0.0;
    unsigned max_iterations = 0;
    double max_time_s = 0.0;
    double start_x = 0.0, start_y = 0.0, start_yaw = 0.0;
    double goal_x = 0.0, goal_y = 0.0;
    std::vector<int> occupancy;
    std::vector<double> probability;
};

struct SamplingStatistics {
    // These counters audit the actual sampling mixture, rather than recording only
    // the configured xy_bias.
    unsigned long long state_sample_calls = 0;
    unsigned long long xy_probability_samples = 0;
    unsigned long long xy_uniform_samples = 0;
};

Input read_input(const char* path) {
    // The input is simple whitespace-delimited text: a scalar header, one occupancy
    // map row, and one probability-map row. This avoids JSON/NumPy dependencies in C++;
    // the fixed kGridSize is also used to validate the number of entries.
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open planner input");
    Input input;
    file >> input.seed >> input.xy_bias >> input.max_iterations >> input.max_time_s
         >> input.start_x >> input.start_y >> input.start_yaw >> input.goal_x >> input.goal_y;
    input.occupancy.resize(kGridSize * kGridSize);
    input.probability.resize(kGridSize * kGridSize);
    for (int& value : input.occupancy) file >> value;
    for (double& value : input.probability) file >> value;
    if (!file) throw std::runtime_error("Planner input is incomplete");
    return input;
}

void write_tree_data(const char* path, const oc::SST& planner,
                     const oc::SpaceInformationPtr& si) {
    // PlannerData exposes the nodes and parent-child connections currently retained
    // by SST. SST has already pruned nodes that lost witness competition, so this is
    // the tree still alive when solve stops, not a log of every historical attempt.
    oc::PlannerData data(si);
    planner.getPlannerData(data);
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Could not open planner tree output");

    file << "{\"vertex_count\":" << data.numVertices()
         << ",\"edge_count\":" << data.numEdges()
         << ",\"vertices\":[";
    for (unsigned index = 0; index < data.numVertices(); ++index) {
        const auto* state = data.getVertex(index).getState()->as<ob::SE2StateSpace::StateType>();
        if (index) file << ',';
        file << '[' << state->getX() << ',' << state->getY() << ',' << state->getYaw() << ']';
    }
    file << "],\"edges\":[";
    bool first_edge = true;
    for (unsigned source = 0; source < data.numVertices(); ++source) {
        std::vector<unsigned> destinations;
        data.getEdges(source, destinations);
        for (const unsigned destination : destinations) {
            if (!first_edge) file << ',';
            file << '[' << source << ',' << destination << ']';
            first_edge = false;
        }
    }
    file << "],\"start_vertices\":[";
    bool first_vertex = true;
    for (unsigned index = 0; index < data.numVertices(); ++index) {
        if (!data.isStartVertex(index)) continue;
        if (!first_vertex) file << ',';
        file << index;
        first_vertex = false;
    }
    file << "],\"goal_vertices\":[";
    first_vertex = true;
    for (unsigned index = 0; index < data.numVertices(); ++index) {
        if (!data.isGoalVertex(index)) continue;
        if (!first_vertex) file << ',';
        file << index;
        first_vertex = false;
    }
    file << "]}\n";
    if (!file) throw std::runtime_error("Failed while writing planner tree output");
}

class XYBiasedSampler final : public ob::StateSampler {
public:
    XYBiasedSampler(const ob::StateSpace* space, const Input& input, SamplingStatistics* statistics)
        : ob::StateSampler(space), input_(input), statistics_(statistics), generator_(input.seed),
          unit_(0.0, 1.0), yaw_(-kPi, kPi) {
        // std::discrete_distribution accepts relative nonnegative weights, so P_xy
        // does not need to be normalized again.
        std::vector<double> weights(input.probability.size());
        double total_weight = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            // The normal P_xy-weighted distribution masks obstacles, even if the model
            // incorrectly assigns them high scores.
            weights[i] = input.occupancy[i] ? 0.0 : std::max(0.0, input.probability[i]);
            total_weight += weights[i];
        }
        // If the model produces an invalid map with all-zero, negative, or obstacle-only
        // weights, fall back to equal weights for every map pixel instead of passing an
        // invalid weight sum to discrete_distribution. This fallback can also sample
        // obstacles; the later state-validity checker will reject those states.
        if (total_weight == 0.0) std::fill(weights.begin(), weights.end(), 1.0);
        distribution_ = std::discrete_distribution<int>(weights.begin(), weights.end());
    }

    void sampleUniform(ob::State* state) override {
        ++statistics_->state_sample_calls;
        auto* sample = state->as<ob::SE2StateSpace::StateType>();
        double x = 0.0, y = 0.0;
        if (unit_(generator_) < input_.xy_bias) {
            ++statistics_->xy_probability_samples;
            const int flat_index = distribution_(generator_);
            // Convert the flat row-major index back to the dataset's (column, row)
            // pixel coordinates.
            const int py = flat_index / kGridSize;
            const int px = flat_index % kGridSize;
            // Sample a subpixel position uniformly inside the selected cell so tree
            // states are not restricted to pixel centers. Image rows increase downward,
            // while world y increases upward, so convert from kMaxM downward.
            x = kMinM + (px + unit_(generator_)) * kResolutionM;
            y = kMaxM - (py + unit_(generator_)) * kResolutionM;
        } else {
            ++statistics_->xy_uniform_samples;
            // The purely uniform branch is the SST baseline at xy_bias=0 and ensures
            // that the mixture retains global exploration.
            x = kMinM + unit_(generator_) * (kMaxM - kMinM);
            y = kMinM + unit_(generator_) * (kMaxM - kMinM);
        }
        sample->setX(x);
        sample->setY(y);
        // This XY-only model deliberately keeps yaw uniform.
        sample->setYaw(yaw_(generator_));
    }

    void sampleUniformNear(ob::State* state, const ob::State* near, double distance) override {
        // Some internal SST/OMPL operations require local sampling. Delegate those
        // requests to the default SE(2) sampler; overriding only sampleUniform
        // preserves the API's standard local-sampling behavior.
        default_sampler_->sampleUniformNear(state, near, distance);
    }

    void sampleGaussian(ob::State* state, const ob::State* mean, double stddev) override {
        // Likewise, Gaussian local perturbations are not modified by P_xy and remain
        // handled by OMPL's default implementation.
        default_sampler_->sampleGaussian(state, mean, stddev);
    }

    void set_default(const ob::StateSamplerPtr& sampler) { default_sampler_ = sampler; }

private:
    const Input& input_;
    SamplingStatistics* statistics_;
    std::mt19937 generator_;
    std::uniform_real_distribution<double> unit_;
    std::uniform_real_distribution<double> yaw_;
    std::discrete_distribution<int> distribution_;
    ob::StateSamplerPtr default_sampler_;
};

class XYGoal final : public ob::GoalRegion {
public:
    XYGoal(const ob::SpaceInformationPtr& si, double x, double y) : ob::GoalRegion(si), x_(x), y_(y) {
        setThreshold(0.20);  // Position-only 20 cm arrival tolerance.
    }

    double distanceGoal(const ob::State* state) const override {
        // GoalRegion compares this continuous distance with threshold. Yaw is
        // intentionally ignored to match the XY-only success definition: reaching the
        // target position is sufficient.
        const auto* sample = state->as<ob::SE2StateSpace::StateType>();
        return std::hypot(sample->getX() - x_, sample->getY() - y_);
    }

private:
    double x_, y_;
};

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3)
            throw std::runtime_error("Usage: NeuralSSTXY <input.txt> [tree.json]");
        const Input input = read_input(argv[1]);
        // Seed OMPL's global random source and independently give the custom sampler
        // the same reproducible experiment seed.
        ompl::RNG::setSeed(input.seed);
        SamplingStatistics sampling_statistics;

        auto space = std::make_shared<ob::SE2StateSpace>();
        // An SE(2) state consists of position (x, y) and yaw; set bounds only on its
        // two-dimensional translational subspace.
        ob::RealVectorBounds bounds(2);
        // A tiny inset prevents floating-point rounding from placing a sample outside
        // the open interval without changing the effective 5 cm grid semantics.
        bounds.setLow(0, kMinM + 1e-9); bounds.setHigh(0, kMaxM - 1e-9);
        bounds.setLow(1, kMinM + 1e-9); bounds.setHigh(1, kMaxM - 1e-9);
        space->setBounds(bounds);
        space->setStateSamplerAllocator([&](const ob::StateSpace* state_space) {
            // OMPL invokes the allocator on demand after setup. References captured to
            // input/statistics remain valid until solve returns, and the sampler's
            // default local sampler is created from the same state_space.
            auto sampler = std::make_shared<XYBiasedSampler>(state_space, input, &sampling_statistics);
            sampler->set_default(state_space->allocDefaultStateSampler());
            return sampler;
        });

        // The sole component of the one-dimensional control space is the front-wheel
        // steering angle δ; speed and reverse controls are not allowed.
        auto controls = std::make_shared<oc::RealVectorControlSpace>(space, 1);
        ob::RealVectorBounds steering_bounds(1);
        steering_bounds.setLow(-kPi / 6.0); steering_bounds.setHigh(kPi / 6.0);
        controls->setBounds(steering_bounds);
        oc::SimpleSetup setup(controls);
        const auto si = setup.getSpaceInformation();
        si->setPropagationStepSize(kPropagationStepS);
        // Each expansion lasts 2--20 propagation steps, forming a constant-steering
        // control segment of 0.04--0.40 s.
        si->setMinMaxControlDuration(2, 20);

        setup.setStateValidityChecker([&](const ob::State* state) {
            const auto* sample = state->as<ob::SE2StateSpace::StateType>();
            // This world-to-pixel transform is the inverse of Python's pixel_to_world.
            // floor consistently assigns cell boundaries to the pixel on their
            // right/bottom; check bounds next to reject all out-of-range states.
            const int px = static_cast<int>(std::floor(sample->getX() / kResolutionM + kGridSize / 2.0));
            const int py = static_cast<int>(std::floor(kGridSize / 2.0 - sample->getY() / kResolutionM));
            return px >= 0 && px < kGridSize && py >= 0 && py < kGridSize &&
                   !input.occupancy[py * kGridSize + px];
        });
        setup.setStatePropagator([](const ob::State* state, const oc::Control* control,
                                    double duration, ob::State* result) {
            const auto* source = state->as<ob::SE2StateSpace::StateType>();
            auto* target = result->as<ob::SE2StateSpace::StateType>();
            const double steering = control->as<oc::RealVectorControlSpace::ControlType>()->values[0];
            // Analytically integrate the kinematic bicycle model under constant speed
            // and steering:
            //   yaw_dot = v/L * tan(δ). The analytic solution avoids accumulated error
            //   from Euler integration at small time steps.
            const double yaw_change = kSpeedMps / kWheelbaseM * std::tan(steering) * duration;
            const double half_change = 0.5 * yaw_change;
            // The sinc form remains continuous as the steering angle approaches zero;
            // directly computing sin(half_change)/half_change would produce 0/0 while
            // driving straight. In that case, distance naturally reduces to
            // v * duration.
            const double sinc = std::abs(half_change) < 1e-9 ? 1.0 : std::sin(half_change) / half_change;
            const double distance = kSpeedMps * duration * sinc;
            const double middle_yaw = source->getYaw() + half_change;
            target->setX(source->getX() + distance * std::cos(middle_yaw));
            target->setY(source->getY() + distance * std::sin(middle_yaw));
            target->setYaw(source->getYaw() + yaw_change);
        });

        // The task specifies the start yaw. XYGoal handles the target, so no final yaw
        // is specified.
        ob::ScopedState<ob::SE2StateSpace> start(space);
        start->setX(input.start_x); start->setY(input.start_y); start->setYaw(input.start_yaw);
        setup.setStartState(start);
        setup.setGoal(std::make_shared<XYGoal>(si, input.goal_x, input.goal_y));
        // Use control duration as the cost, but set the threshold to positive infinity
        // so optimization stops when the first exact solution is found. This measures
        // first-solution performance rather than continued path optimization within
        // the budget.
        auto objective = std::make_shared<ob::ControlDurationObjective>(si);
        objective->setCostThreshold(ob::Cost(std::numeric_limits<double>::infinity()));
        setup.setOptimizationObjective(objective);
        auto planner = std::make_shared<oc::SST>(si);
        // Disable SST's own direct goal sampling so XY guidance comes only from the
        // explicit P_xy mixture sampler.
        planner->setGoalBias(0.0);
        // The selection radius defines the local region for representative-node
        // competition. A smaller pruning radius removes higher-cost nearby nodes,
        // limiting tree size while preserving coverage of distinct reachable
        // kinematic directions.
        planner->setSelectionRadius(0.50);
        planner->setPruningRadius(0.10);
        setup.setPlanner(planner);

        const auto began = std::chrono::steady_clock::now();
        unsigned termination_calls = 0;
        ob::PlannerTerminationCondition termination([&]() {
            ++termination_calls;
            // OMPL queries this predicate repeatedly. The call count is intentionally
            // used as a deterministic approximate work budget, not as a strict synonym
            // for "number of expansions." The wall-clock limit also protects against
            // slow machines and pathological tasks.
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            return termination_calls > input.max_iterations || elapsed >= input.max_time_s;
        });
        // solve may return an exact solution, an approximate solution, or no solution;
        // only exact solutions count as success below.
        setup.solve(termination);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        const bool success = setup.haveExactSolutionPath();
        if (argc == 3) write_tree_data(argv[2], *planner, si);
        // Emit machine-readable JSON strictly on the last line for run.py to parse.
        // Even on failure, include complete empty sequences and statistics so batch
        // processing does not have to infer failure from log text.
        std::cout << "{\"success\":" << (success ? "true" : "false")
                  << ",\"solution_level\":\"" << (success ? "exact" : "failed") << "\""
                  << ",\"termination_reason\":\"" << (success ? "exact_solution" :
                      (termination_calls > input.max_iterations ? "iteration_limit" : "time_limit")) << "\""
                  << ",\"planning_time_s\":" << elapsed
                  << ",\"first_solution_iterations\":" << termination_calls
                  << ",\"state_sample_calls\":" << sampling_statistics.state_sample_calls
                  << ",\"xy_probability_samples\":" << sampling_statistics.xy_probability_samples
                  << ",\"xy_uniform_samples\":" << sampling_statistics.xy_uniform_samples
                  << ",\"state_sequence\":[";
        if (setup.haveSolutionPath()) {
            // Path states are the discrete endpoints of control segments, each encoded
            // as [x_m, y_m, yaw_rad].
            const auto path = setup.getSolutionPath();
            for (unsigned index = 0; index < path.getStateCount(); ++index) {
                const auto* state = path.getState(index)->as<ob::SE2StateSpace::StateType>();
                if (index) std::cout << ',';
                std::cout << '[' << state->getX() << ',' << state->getY() << ',' << state->getYaw() << ']';
            }
        }
        std::cout << "],\"control_sequence\":[";
        if (setup.haveSolutionPath()) {
            // Keep a separate steering-angle array for compatibility with earlier
            // consumers; the next field contains the more complete duration data.
            const auto path = setup.getSolutionPath();
            for (unsigned index = 0; index < path.getControlCount(); ++index) {
                if (index) std::cout << ',';
                std::cout << path.getControl(index)->as<oc::RealVectorControlSpace::ControlType>()->values[0];
            }
        }
        std::cout << "],\"control_segments\":[";
        if (setup.haveSolutionPath()) {
            // Save control_segments alongside state_sequence so Python can reconstruct
            // trajectory timing and total control duration, not just the geometric
            // polyline.
            const auto path = setup.getSolutionPath();
            for (unsigned index = 0; index < path.getControlCount(); ++index) {
                if (index) std::cout << ',';
                const double steering = path.getControl(index)->as<oc::RealVectorControlSpace::ControlType>()->values[0];
                std::cout << "{\"steering_rad\":" << steering
                          << ",\"duration_s\":" << path.getControlDuration(index) << '}';
            }
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        // Ensure malformed input, missing files, or OMPL setup errors are reported to
        // Python's subprocess.run(check=True) with a nonzero exit code, rather than
        // masquerading as a parseable planning-failure JSON result.
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
