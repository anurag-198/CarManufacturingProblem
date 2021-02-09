import copy
import operator


class Search_node():
    def __init__(self, env, max_depth, value=0, depth=0, visited=[]):
        self.env = env
        self.visited = visited
        self.value = value
        self.done = env.done
        self.depth = depth
        self.max_depth = max_depth

    def explore(self):
        res = []
        for i in self.env.possible_actions():
            child_visited = copy.deepcopy(self.visited)
            child_visited.append(i)
            child_env, (reward, next_state, done) = self.env.light_step(i)
            child = Search_node(child_env, self.max_depth, self.value + reward, self.depth + 1, child_visited)
            res.append(child)

        return res

    def show(self):
        print(self.value, self.visited)

    def terminal(self):
        return (self.done or self.depth == self.max_depth)

class Look_ahead_search():

    def __init__(self, depth = 10):
        self.depth = depth

    def act(self, env):
        result = None
        root = Search_node(env, self.depth)
        open_list = [root]
        while len(open_list) != 0:
            actual = open_list.pop()
            if actual.terminal():
                result = actual
                break
            children = actual.explore()
            for child in children:
                open_list.append(child)
            open_list.sort(key=operator.attrgetter('value'))

        return result.visited[0]