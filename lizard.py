class Lizard:
    def __init__(self, name):
        self.name = name

    def set_name(self, name):
        self.name = name


l = Lizard('deep')
print(l.name)
