"""
A calibrated, pool-by-pool simulator of a thoroughbred racing circuit.

- params       every knob, with public defaults and an optional calibrated JSON override
- population   horses, trainers, jockeys, owners and sires with hidden qualities
- racing       cards, entries, the running of a race and careers
- tote         crowd beliefs, money arrival cycle by cycle, win / place / show and exotic payoffs
- horizontals  daily doubles (matrices and will-pays) and Pick-N pools with carryovers
- simulate     the day-by-day orchestrator and conversion to the ``lynk.horse_racing.schema`` frame
- fidelity     the stylized facts used to compare simulated and real racing
"""

from lynk.horse_racing.simulator.params import load_params, save_params, DEFAULTS  # noqa: F401
from lynk.horse_racing.simulator.simulate import simulate, to_schema_frame, SimulationResult  # noqa: F401
