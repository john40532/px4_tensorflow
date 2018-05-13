import numpy as np
import math
import time
import threading

class Controller_PID_Point2Point():
    def __init__(self, params):
        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0]/180.0)*3.14,(params['Tilt_limits'][1]/180.0)*3.14]
        self.YAW_CONTROL_LIMITS = params['Yaw_Control_Limits']
        self.Z_LIMITS = [-params['Z_XY_offset'],params['Z_XY_offset']]
        self.LINEAR_P = params['Linear_PID']['P']
        self.LINEAR_I = params['Linear_PID']['I']
        self.LINEAR_D = params['Linear_PID']['D']
        self.linear_anti_windup = np.array([0,0,0])
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
        self.ANGULAR_P = params['Angular_PID']['P']
        self.ANGULAR_I = params['Angular_PID']['I']
        self.ANGULAR_D = params['Angular_PID']['D']
        self.angular_anti_windup = np.array([0,0,0])
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.thread_object = None
        self.yaw_target = 0.0
        self.run = True

    def wrap_angle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def quaternion_to_euler_angle(self, w, x, y, z):
        ysqr = y * y
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = math.atan2(t3, t4)
        
        return X, Y, Z

    def update(self, target, state):
        [dest_x,dest_y,dest_z] = target
        [x,y,z,qx,qy,qz,qw,x_dot,y_dot,z_dot,theta_dot,phi_dot,gamma_dot] = state
        x_error = dest_x-x
        y_error = dest_y-y
        x_error = np.clip(x_error,-1,1)
        y_error = np.clip(y_error,-1,1)

        z_error = dest_z-z
        theta, phi, gamma = self.quaternion_to_euler_angle(qw, qx, qy, qz)
        self.xi_term = self.LINEAR_I[0]*x_error - self.linear_anti_windup[0]
        self.yi_term = self.LINEAR_I[1]*y_error - self.linear_anti_windup[1]
        self.zi_term = self.LINEAR_I[2]*z_error - self.linear_anti_windup[2]
        
        dest_x_dot = self.LINEAR_P[0]*(x_error) + self.LINEAR_D[0]*(-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1]*(y_error) + self.LINEAR_D[1]*(-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2]*(z_error) + self.LINEAR_D[2]*(-z_dot) + self.zi_term
        
        throttle_a_t = np.clip(dest_z_dot,self.Z_LIMITS[0],self.Z_LIMITS[1])
        self.linear_anti_windup[2] = dest_z_dot - throttle_a_t

        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*math.sin(gamma)-dest_y_dot*math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*math.cos(gamma)+dest_y_dot*math.sin(gamma))
        dest_gamma = self.yaw_target
        
        dest_theta = np.clip(dest_theta,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
        dest_phi = np.clip(dest_phi,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
        theta_error = dest_theta-theta
        phi_error = dest_phi-phi
        # theta_error = 0-theta
        # phi_error = 0-phi
        gamma_dot_error = (self.YAW_RATE_SCALER*self.wrap_angle(dest_gamma-gamma)) - gamma_dot
        
        self.thetai_term = self.ANGULAR_I[0]*theta_error - self.angular_anti_windup[0]
        self.phii_term = self.ANGULAR_I[1]*phi_error - self.angular_anti_windup[1]
        self.gammai_term += self.ANGULAR_I[2]*gamma_dot_error

        x_val = self.ANGULAR_P[0]*(theta_error) + self.ANGULAR_D[0]*(-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1]*(phi_error) + self.ANGULAR_D[1]*(-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2]*(gamma_dot_error) + self.gammai_term
        x_val_a_t = np.clip(x_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])
        y_val_a_t = np.clip(y_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])
        self.angular_anti_windup[0] = x_val - x_val_a_t
        self.angular_anti_windup[1] = y_val - y_val_a_t
        z_val = np.clip(z_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])

        m1 = throttle_a_t - 0.5*x_val_a_t - 0.5*y_val_a_t - z_val + 550
        m2 = throttle_a_t + 0.5*x_val_a_t + 0.5*y_val_a_t - z_val + 550
        m3 = throttle_a_t + 0.5*x_val_a_t - 0.5*y_val_a_t + z_val + 550
        m4 = throttle_a_t - 0.5*x_val_a_t + 0.5*y_val_a_t + z_val + 550
        M = np.clip([m1,m2,m3,m4],self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])
        return M

    def updatePD(self, target, state):
        [dest_x,dest_y,dest_z] = target
        [x,y,z,qx,qy,qz,qw,x_dot,y_dot,z_dot,theta_dot,phi_dot,gamma_dot] = state
        x_error = dest_x-x
        y_error = dest_y-y
        x_error = np.clip(x_error,-2,2)
        y_error = np.clip(y_error,-2,2)

        z_error = dest_z-z
        theta, phi, gamma = self.quaternion_to_euler_angle(qw, qx, qy, qz)
        
        dest_x_dot = self.LINEAR_P[0]*(x_error) + self.LINEAR_D[0]*(-x_dot)
        dest_y_dot = self.LINEAR_P[1]*(y_error) + self.LINEAR_D[1]*(-y_dot)
        dest_z_dot = self.LINEAR_P[2]*(z_error) + self.LINEAR_D[2]*(-z_dot)
        
        throttle = np.clip(dest_z_dot,self.Z_LIMITS[0],self.Z_LIMITS[1])

        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*math.sin(gamma)-dest_y_dot*math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*math.cos(gamma)+dest_y_dot*math.sin(gamma))
        dest_gamma = self.yaw_target
        
        dest_theta = np.clip(dest_theta,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
        dest_phi = np.clip(dest_phi,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
        theta_error = dest_theta-theta
        phi_error = dest_phi-phi
        gamma_dot_error = (self.YAW_RATE_SCALER*self.wrap_angle(dest_gamma-gamma)) - gamma_dot

        x_val = self.ANGULAR_P[0]*(theta_error) + self.ANGULAR_D[0]*(-theta_dot)
        y_val = self.ANGULAR_P[1]*(phi_error) + self.ANGULAR_D[1]*(-phi_dot)
        z_val = self.ANGULAR_P[2]*(gamma_dot_error)
        x_val = np.clip(x_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])
        y_val = np.clip(y_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])
        z_val = np.clip(z_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])

        m1 = throttle - 0.5*x_val - 0.5*y_val - z_val + 600
        m2 = throttle + 0.5*x_val + 0.5*y_val - z_val + 600
        m3 = throttle + 0.5*x_val - 0.5*y_val + z_val + 600
        m4 = throttle - 0.5*x_val + 0.5*y_val + z_val + 600
        M = np.clip(np.array([m1,m2,m3,m4]),self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])
        return M

