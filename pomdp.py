import random

class Hallway:
    def __init__(self, p_forward = 0.95):
        self.p_forward = p_forward
        self.dir = True

    def __iter__(self):
        while True:
            action = random.random() < self.p_forward
            yield action
            if not action: # Rotation
                self.dir = not self.dir
            yield self.dir

class Room:
    def __init__(self, p_forward = 0.95):
        self.p_forward = p_forward
        self.dir = True
        self.dist_to_wall = 0

    def __iter__(self):
        while True:
            action = random.random() < self.p_forward
            yield action
            if not action: # Rotation
                self.dir = not self.dir
            else: # Forward
                self.dist_to_wall += (1 if self.dir else -1)
            yield self.dist_to_wall == 0 and self.dir == 0


def Switcher(switch_prob = 0.02, *modes):
    m = iter(random.choice(modes))
    while True:
        if random.random() < switch_prob:
            m = iter(random.choice(modes))
        yield m.send(None)
        yield m.send(None)
        
    
    
