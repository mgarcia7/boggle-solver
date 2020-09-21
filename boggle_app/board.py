class Trie:
    def __init__(self):
        self.children = {}
        self.endFlag = False # set true if it's the end of a word (even if there can be more nodes after it)

    def contains(self,word):
        # returns (isPrefix,isfullword)
        word = word.upper()
        current = self
        prefix = False
        for ch in word:
            if ch not in current.children:
                current = Trie() # makes current empty
                break
            current = current.children.get(ch)
        else: # if it gets through the whole word without breaking, then it must be a prefix to a word
            prefix = True

        return (prefix, current.endFlag)

    def insert(root, word):
        current = root
        for ch in word:
            if ch not in current.children:
                current.children[ch] = Trie()
            current = current.children[ch]
        current.endFlag = True

def load_dictionary(filename):
    root = Trie()

    with open(filename) as f:
        for line in f:
            root.insert(line.strip())

    return root

class Node:
    def __init__(self,value=[],neighbors=[],coords=(-1,-1)):
        self.value = value
        self.neighbors = neighbors
        self.visited = False
        self.coords = coords

    def __repr__(self):
        return "<Node {}:{},{}>".format(self.value,self.coords[0],self.coords[1])

class Board:
    def __init__(self,size_x,size_y,letters,D):
        self.size_x = size_x
        self.size_y = size_y
        self.letters = letters
        self.D = D
        self.nodes = []
        self.valid_words = set()

    def find_neighbors_indices(self,coords):
        x,y = coords
        valid_coords = lambda x,y: x>=0 and y>=0 and x<self.size_x and y<self.size_y
        n = [(x-1, y-1),(x, y-1),(x+1,y-1),(x-1,y),(x+1,y),\
                (x-1,y+1),(x,y+1),(x+1,y+1)]
        return [(x,y) for x,y in n if valid_coords(x,y)]

    def get_letter(self,coords):
        return self.letters[coords[0]][coords[1]]

    def mat2lin(self,i,j):
        return i*self.size_y + j

    def dfs(self,src,stack):
        src.visited = True
        stack.append(src.value)

        word = ''.join(stack)
        isPrefix,isWord = self.D.contains(word)
#         print('{}: ({},{})'.format(word,isPrefix,isWord))
        if isWord and len(word) > 3:
            self.valid_words.add(word)

        if isPrefix:
            # add neighbors to stack only if it is a prefix
            for node in src.neighbors:
                if not node.visited:
                    self.dfs(node,stack)

        stack.pop()
        src.visited = False


    def search(self):
        # replace letters with nodes
        nodes = [0 for _ in range(self.size_x*self.size_y)]
        for x in range(self.size_x):
            for y in range(self.size_y):
                nodes[self.mat2lin(x,y)] = Node(value=self.letters[x][y],\
                                       coords = (x,y))

        # calculate neighbors
        for n in nodes:
            n.neighbors = [nodes[self.mat2lin(x,y)] for x,y in self.find_neighbors_indices(n.coords)]

        # start looking for paths from every node
        for src_node in nodes:
            self.dfs(src_node,[])

    def print_words(self):
        # sort by length
        for word in self.get_words():
            print(word)


    def get_words(self):
        return sorted(list(self.valid_words),key=lambda x: len(x),reverse=True)
