class ReflexBot:
    def __init__(self, env):
        self.env = env

    def select_action(self):
        if self.env.is_valid_combo():
            return 1
        return 0
