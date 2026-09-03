"""Optional keyboard control for interactive policy evaluation."""

try:
    from pynput import keyboard
except ImportError:  # Optional outside interactive evaluation.
    keyboard = None


class KeyboardController:
    """Update time ratio and optional configuration index from arrow keys."""

    def __init__(self, envs, start_value=1.0, start_config=None):
        self.envs = envs
        self.current_value = start_value
        self.config_idx = (
            start_config
            if start_config is not None
            else self.envs.cfg.get("specific_idx")
        )
        self.sr_range = (0.2, 1.0)
        self.config_range = (0, 999)
        self.scevelSchedule = self.envs.cfg.get("scevelSchedule", 1.0)
        self.running = True
        self.listener = None
        self.last_key_pressed = None
        self.key_flash = {"UP": False, "DOWN": False}

    def start(self):
        if keyboard is None:
            raise RuntimeError(
                "Keyboard control requires pynput and a working display session."
            )
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()

    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()

    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.current_value += 0.1
                self.clip_current_value()
                self.last_key_pressed = "UP"
                self.key_flash["UP"] = True
                self.env_callback()
            elif key == keyboard.Key.down:
                self.current_value -= 0.1
                self.clip_current_value()
                self.last_key_pressed = "DOWN"
                self.key_flash["DOWN"] = True
                self.env_callback()
            elif key == keyboard.Key.left and self.config_idx is not None:
                self.config_idx -= 1
                self.clip_current_value()
                self.env_callback()
            elif key == keyboard.Key.right and self.config_idx is not None:
                self.config_idx += 1
                self.clip_current_value()
                self.env_callback()
            elif getattr(key, "char", None) == "s":
                print(
                    f"Current config idx: {self.config_idx}, "
                    f"Current speed ratio: {self.current_value}"
                )
        except (AttributeError, TypeError):
            return

    def clip_current_value(self):
        self.current_value = max(
            min(self.current_value, self.sr_range[1]), self.sr_range[0]
        )
        if self.config_idx is not None:
            self.config_idx = max(
                min(self.config_idx, self.config_range[1]), self.config_range[0]
            )

    def reset_key_flash(self):
        self.key_flash["UP"] = False
        self.key_flash["DOWN"] = False

    def env_callback(self):
        self.envs.goal_speed = self.current_value
        self.envs.update_time_ratio_buf(self.current_value)
        self.envs.update_linvel_gt()
        self.envs.cfg["specific_idx"] = self.config_idx
