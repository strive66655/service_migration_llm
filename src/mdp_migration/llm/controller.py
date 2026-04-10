from __future__ import annotations

from ..core import CostParams
from .schema import SafeControlParams


def apply_control_params(base_cost_params: CostParams, control_params: SafeControlParams) -> CostParams:
    migration_weight = control_params.migration_weight
    transmission_weight = control_params.transmission_weight
    return CostParams(
        gamma=control_params.gamma,
        power_factor=base_cost_params.power_factor,
        const_factor_migrate=base_cost_params.const_factor_migrate * migration_weight,
        proportional_factor_migrate=base_cost_params.proportional_factor_migrate * migration_weight,
        const_factor_trans=base_cost_params.const_factor_trans * transmission_weight,
        proportional_factor_trans=base_cost_params.proportional_factor_trans * transmission_weight,
    )
