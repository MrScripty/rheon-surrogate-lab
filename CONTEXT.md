# Rheon pressure surrogate experiments

Shared language for comparing learned pressure solvers, numerical solvers and
their effects on time-evolving fluid simulations.

## Language

### Space and material

**Grid dimensions**:
The number of cells along each axis, such as 32 × 24 × 16.
_Avoid_: Domain size when it is unclear whether counts or metres are meant.

**Cell count**:
The total number of cells, equal to the product of the grid dimensions.

**Grid resolution**:
The number of cells per metre; for cubic cells it is the reciprocal of cell spacing.
_Avoid_: Density without specifying that it means cells per metre.

**Cell spacing**:
The physical width of a cell in metres. Cubic cells have the same spacing along
all three axes.

**Physical extent**:
The domain lengths in metres, obtained by multiplying grid dimensions by cell
spacing.
_Avoid_: Resolution when referring to the physical dimensions of the domain.

**Fluid density**:
Mass per unit fluid volume, measured in kilograms per cubic metre.
_Avoid_: Grid density.

### Time and solver work

**Simulation timestep**:
The amount of physical simulated time between successive simulation states.
_Avoid_: Inference time when referring to simulated time.

**Pressure correction pass**:
One application of a pressure correction within a single pressure request.
_Avoid_: Simulation step when no physical time has advanced.

**Inference time**:
Elapsed computation time for a specified prediction procedure, with its included
operations stated.

### Accuracy and comparison

**Equation residual**:
The difference between a pressure equation's right-hand side and its discrete
operator applied to the proposed pressure.

**Same-request pressure error**:
The difference between predicted and reference pressure for the same pressure
equation and input state.

**Trajectory drift**:
The difference between corresponding fields in independently advanced trajectories
at matching physical times, from matched initial conditions and forcing.
_Avoid_: Pressure residual when describing accumulated trajectory differences.

**Accuracy and time Pareto frontier**:
The measured solver configurations for which no other compared configuration has
both no worse accuracy and no greater runtime, with at least one strict improvement.
Comparisons share an evaluation population, accuracy definition and timing scope.
