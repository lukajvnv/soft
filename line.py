class Line:
    def __init__(self, line_array):
        x1, y1, x2, y2 = line_array
        self._first_dot = (x1, y1)
        self._last_dot = (x2, y2)

    def first_dot(self):
        return self._first_dot

    def last_dot(self):
        return self._last_dot
