from matplotlib import pyplot as plt
import random
import numpy as np
import queue
import math
class SimplePID:
    def __init__(self, kp, ki, kd, dead_zone=None,integration_seprate = None ,integration_bound=None, alpha=0.0,output_limits=(-10000, 10000)):

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits
        self.err_sum = 0
        self.alpha = alpha
        
        self.integration_bound_lower,self.integration_bound_upper = integration_bound
    
        self.integration_seprate = integration_seprate
    
        self.dead_zone = dead_zone

        # 内部状态
        self._prev_error = 0.0  # 上次误差

    def compute(self, err, derr=None):
        """
        计算控制输出

        参数:
        current: 当前测量值

        返回:
        限幅后的控制输出
        """
        # 死区处理
        if self.dead_zone is not None and np.fabs(err) < self.dead_zone:
            err = 0

        # 积分分离
        if self.integration_seprate is not None and np.fabs(err) > self.integration_seprate:
            self.err_sum = 0

        # pid计算
        # 这里的err是当前误差，无derr使用上一次的控制量
        if derr is None:
            err_filtered = err*(1-self.alpha) + self._prev_error*self.alpha
            self.err_sum += err_filtered

            # 积分限幅
            if self.integration_bound_upper is not None and self.err_sum > self.integration_bound_upper:
                self.err_sum = self.integration_bound_upper
            elif self.integration_bound_lower is not None and self.err_sum < self.integration_bound_lower:
                self.err_sum = self.integration_bound_lower

            output = self.kp*err_filtered + self.ki * self.err_sum + self.kd*(err_filtered - self._prev_error)
        
            self._prev_error = err_filtered
            
        #微分先行
        else:
            self.err_sum += err
            # 积分限幅
            if self.integration_bound_upper is not None and self.err_sum > self.integration_bound_upper:
                self.err_sum = self.integration_bound_upper
            elif self.integration_bound_lower is not None and self.err_sum < self.integration_bound_lower:
                self.err_sum = self.integration_bound_lower
            output = self.kp*err + self.ki * self.err_sum + self.kd*derr
            self._prev_error = err

        output = max(self.min_out,min(self.max_out, output))

        return output

    def reset(self,**kwargs):
        """
        重置PID控制器的参数和状态
        """
        self.err_sum = 0
        self._prev_error = 0.0
        self.integration_bound_lower,self.integration_bound_upper = kwargs.get('integration_bound',(self.integration_bound_lower,self.integration_bound_upper))
        self.integration_seprate = kwargs.get('integration_seprate',self.integration_seprate)
        self.dead_zone = kwargs.get('dead_zone',self.dead_zone)
        self.alpha = kwargs.get('alpha',self.alpha)
        self.kp = kwargs.get('kp',self.kp)
        self.ki = kwargs.get('ki',self.ki)
        self.kd = kwargs.get('kd',self.kd)
        self.min_out,self.max_out = kwargs.get('output_limits',(self.min_out,self.max_out))
    
    def get_pid(self):
        return (self.kp, self.ki, self.kd)


#if __name__ == "__main__":
    # 示例：使用SimplePID控制器进行模拟
    # 这里的代码是一个简单的PID控制器模拟示例，您可以根据需要修改参数和逻辑
    # # PID参数设置
    # kp = 80
    # ki = 1.0
    # kd = 200.0
    # TARGET = 50.0
    # OUTPUT_LIMITS = (-100, 100)  # 输出限幅

    # # 初始化PID控制器
    # pid = SimplePID(kp, ki, kd, dead_zone=0.1)

    # # 模拟参数
    # dt = 0.1  # 时间步长（秒）
    # total_time = 30  # 总模拟时间（秒）
    # steps = int(total_time / dt)
    # current_value = 0.0  # 系统初始状态

    # # 数据记录列表
    # time_points = []
    # system_values = []
    # control_outputs = []

    # # 模拟循环
    # for step in range(steps):
    #     # 计算控制输出
    #     print(current_value)
    #     output = pid.compute(TARGET-current_value)
        
    #     # 模拟系统动态（示例：一阶惯性系统）
    #     # 系统方程：current_value += (output - current_value) * 0.05 * dt
    #     current_value += (output - current_value) * 0.05 * dt + random.gauss(0.0,0.1)
        
    #     # 在第10秒添加干扰
    #     if step * dt == 10.0:
    #         current_value += 0
        
    #     # 记录数据
    #     time_points.append(step * dt)
    #     system_values.append(current_value)
    #     control_outputs.append(output)

    # # 可视化结果
    # plt.figure(figsize=(12, 8))

    # # 系统响应曲线
    # plt.subplot(2, 1, 1)
    # plt.plot(time_points, system_values, label='System Response')
    # plt.plot(time_points, [TARGET]*len(time_points), 'r--', label='Target')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Value')
    # plt.title('PID Controller Performance')
    # plt.legend()
    # plt.grid(True)

    # # 控制输出曲线
    # plt.subplot(2, 1, 2)
    # plt.plot(time_points, control_outputs, 'g', label='Control Output')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Output')
    # plt.title('Control Signal')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()