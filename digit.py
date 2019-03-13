class Digit:
    def __init__(self, x, y, w, h):
        self.top_l = (x, y)
        self.bottom_l = (x, y + h)
        self.top_r = (x + w, y)
        self.bottom_r = (x + w, y + h)

        self.detect_value = -1
        self.is_detected = False

        self.passed_line = False
        self.passed_blue_line = False
        self.passed_green_line = False

    @property
    def top_l(self):
        return self.__top_l

    @top_l.setter
    def top_l(self, t_l):
        self.__top_l = t_l

    @property
    def bottom_l(self):
        return self.__bottom_l

    @bottom_l.setter
    def bottom_l(self, b_l):
        self.__bottom_l = b_l

    @property
    def top_r(self):
        return self.__top_r

    @top_r.setter
    def top_r(self, t_r):
        self.__top_r = t_r

    @property
    def bottom_r(self):
        return self.__bottom_r

    @bottom_r.setter
    def bottom_r(self, b_r):
        self.__bottom_r = b_r

    def update_coordinates(self, recent_coord_obj):
        self.top_r = recent_coord_obj.top_r
        self.top_l = recent_coord_obj.top_l
        self.bottom_l = recent_coord_obj.bottom_l
        self.bottom_r = recent_coord_obj.bottom_r

    def between_line(self, line):
        (x1, y1) = line.first_dot()
        (x2, y2) = line.last_dot()
        (xnum, ynum) = self.bottom_r

        return x1 <= xnum <= x2 and y2 <= ynum <= y1
