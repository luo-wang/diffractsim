    改变一下高斯滤波参数，结果更符合实验值

        elif method == "A":
            sigma_x = self.pixelperiodx / (0.63 * self.dx)  # 一半像素周期
            sigma_y = self.pixelperiody / (0.63 * self.dy)