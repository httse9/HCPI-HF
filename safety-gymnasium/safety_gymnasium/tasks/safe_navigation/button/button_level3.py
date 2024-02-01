
"""Button task 1."""

from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.mocaps import Gremlins
from safety_gymnasium.tasks.safe_navigation.button.button_level0 import ButtonLevel0


class ButtonLevel3(ButtonLevel0):
    """An agent must press a goal button while avoiding hazards and gremlins.

    And while not pressing any of the wrong buttons.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Hazards(num=4, keepout=0.18))
        self.buttons.is_constrained = True  # pylint: disable=no-member
