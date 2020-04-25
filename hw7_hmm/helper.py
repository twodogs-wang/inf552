class Node:
    def __init__(self, pre = None, prob = 0.0,c=(0,0)):
        self.pre = pre
        #self.next = n
        self.prob = prob
        self.coordinates=c

    def get_coordinates(self):
        return self.coordinates
