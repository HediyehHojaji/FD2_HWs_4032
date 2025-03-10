#سوال سه
import numpy as np

def analyze_transform_matrix(C):
    """
    ورودی: C یک ماتریس 3x3
    خروجی:
      - اگر C یک ماتریس انتقال معتبر نباشد، یک پیام متنی برمی‌گرداند.
      - اگر معتبر باشد، دیکشنری شامل سه کلید زیر برمی‌گرداند:
         1) 'Euler_angles'      : لیستی از زوایای [roll, pitch, yaw] بر حسب درجه
         2) 'Rotation_vector'   : لیستی از [rx, ry, rz] که بردار دوران است
         3) 'Quaternion_vector' : لیستی از [q0, q1, q2, q3]
    """

    # 1) بررسی خاصیت متعامد بودن: C * C^T = I
    identity_check = np.allclose(C @ C.T, np.eye(3), atol=1e-6, rtol=1e-6)

    # 2) بررسی مقدار دترمینان باید نزدیک به ۱ باشد
    det_C = np.linalg.det(C)
    det_check = np.isclose(det_C, 1.0, atol=1e-6, rtol=1e-6)

    # اگر یکی از این دو شرط برقرار نباشد، ماتریس انتقال معتبر نیست
    if not identity_check or not det_check:
        return "ماتريس ورودى شرايط يك ماتريس انتقال را ندارد."

    # ----------------------------
    # اگر ماتریس انتقال معتبر است:
    # ----------------------------

    # --- الف) محاسبه زوایای اویلر (Euler Angles) ---
    # توجه: این فرمول‌ها بر اساس یک قرارداد خاص (مثلاً Z-Y-X) هستند
    # اینجا فرض می‌کنیم ترتیب Z-Y-X برای محاسبه رول، پیچ، یاو استفاده شود
    # یکی از روابط متداول:
    roll  = np.arctan2(C[2,1], C[2,2])         # φ
    pitch = np.arcsin(-C[2,0])                # θ
    yaw   = np.arctan2(C[1,0], C[0,0])        # ψ

    # تبدیل به درجه
    roll_deg  = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg   = np.degrees(yaw)

    # --- ب) محاسبه بردار دوران (Rotation Vector) ---
    # زاویه دوران
    theta_r = np.arccos((np.trace(C) - 1) / 2)
    # اگر θ_r بسیار کوچک باشد، ممکن است تقسیم بر صفر رخ دهد
    # در عمل باید بررسی کنیم که sin(theta_r) صفر نباشد
    eps = 1e-12
    if abs(np.sin(theta_r)) < eps:
        # در این حالت، چرخش نزدیک به صفر است
        rotation_vec = [0.0, 0.0, 0.0]
    else:
        rx = (C[2,1] - C[1,2]) / (2 * np.sin(theta_r))
        ry = (C[0,2] - C[2,0]) / (2 * np.sin(theta_r))
        rz = (C[1,0] - C[0,1]) / (2 * np.sin(theta_r))
        rotation_vec = [rx, ry, rz]

    # --- پ) محاسبه کواترنیون (Quaternion) ---
    # از زوایای اویلر به دست آمده در بالا (roll, pitch, yaw) برای ساخت کواترنیون استفاده می‌کنیم
    # فرمول استاندارد (ترتیب چرخش Z-Y-X)
    half_roll  = roll  / 2
    half_pitch = pitch / 2
    half_yaw   = yaw   / 2

    c1 = np.cos(half_roll)
    c2 = np.cos(half_pitch)
    c3 = np.cos(half_yaw)
    s1 = np.sin(half_roll)
    s2 = np.sin(half_pitch)
    s3 = np.sin(half_yaw)

    q0 = c1*c2*c3 + s1*s2*s3  # اسکالر
    q1 = s1*c2*c3 - c1*s2*s3
    q2 = c1*s2*c3 + s1*c2*s3
    q3 = c1*c2*s3 - s1*s2*c3

    quaternion_vec = [q0, q1, q2, q3]

    # --- ت) خروجی نهایی در قالب دیکشنری ---
    return {
        "Euler_angles"     : [roll_deg, pitch_deg, yaw_deg],
        "Rotation_vector"  : rotation_vec,
        "Quaternion_vector": quaternion_vec
    }

# ------------------------------
# نمونه استفاده از تابع بالا
if name == "__main__":
    # یک ماتریس مثال (همان ماتریس سوال ۳)
    C_example = np.array([
        [0.2802, 0.1387, 0.9499],
        [0.1962, 0.9603, -0.1981],
        [-0.9397, 0.2418, 0.2418]
    ])

    result = analyze_transform_matrix(C_example)
    print(result)

    #سوال دوم
    import numpy as np
    import math


    def compute_euler_data(angular_velocity):
        """
        این تابع بردار سرعت زاویه‌ای هواپیما را به صورت یک لیست دریافت می‌کند
        و یک دیکشنری با دو کلید زیر برمی‌گرداند:
          - 'Euler_angles': لیستی از زاویه‌های اویلر [رول, پیچ, یاو] (به درجه)
          - 'Euler_angles_rate': لیستی از نرخ تغییرات این زاویه‌ها (به رادیان بر ثانیه)

        در اینجا برای مسئله ۲، فرض می‌شود:
          - زاویه رول (phi) = 60 درجه
          - زاویه پیچ (theta) = 0 درجه
          - زاویه یاو (psi) = 0 درجه (فرض اولیه)

        برای محاسبه نرخ تغییرات زاویه‌ای از ماتریس تبدیل اویلر استفاده می‌کنیم:

          [ω_x]   [ 1        sin(phi)*tan(theta)   cos(phi)*tan(theta) ]   [  φ̇  ]
          [ω_y] = [ 0            cos(phi)             -sin(phi)         ] * [  θ̇  ]
          [ω_z]   [ 0        sin(phi)/cos(theta)    cos(phi)/cos(theta) ]   [  ψ̇  ]

        سپس با معکوس‌گیری از این ماتریس، نرخ تغییرات اویلر (φ̇, θ̇, ψ̇) را به دست می‌آوریم.
        """
        # داده‌های مسئله: زاویه‌های اویلر به صورت اولیه (درجه)
        phi = 60.0  # رول (درجه)
        theta = 0.0  # پیچ (درجه)
        psi = 0.0  # یاو (درجه)؛ فرض اولیه

        # تبدیل زاویه‌های رول و پیچ به رادیان
        phi_rad = math.radians(phi)
        theta_rad = math.radians(theta)

        # تعریف ماتریس تبدیل اویلر (T) بر اساس زوایای phi و theta
        T_euler = np.array([
            [1, math.sin(phi_rad) * math.tan(theta_rad), math.cos(phi_rad) * math.tan(theta_rad)],
            [0, math.cos(phi_rad), -math.sin(phi_rad)],
            [0, math.sin(phi_rad) / math.cos(theta_rad), math.cos(phi_rad) / math.cos(theta_rad)]
        ])

        # تبدیل بردار سرعت زاویه‌ای ورودی به آرایه numpy
        omega = np.array(angular_velocity)

        # محاسبه نرخ تغییرات اویلر با حل معادله: T * [φ̇, θ̇, ψ̇]^T = ω
        euler_rates = np.linalg.solve(T_euler, omega)

        # آماده‌سازی خروجی به صورت دیکشنری
        result = {
            "Euler_angles": [phi, theta, psi],  # زاویه‌های اویلر به درجه
            "Euler_angles_rate": euler_rates.tolist()  # نرخ تغییرات اویلر به رادیان بر ثانیه
        }

        return result


    # نمونه استفاده:
    angular_velocity_vector = [0.33, 0.28, 0.16]  # ورودی: [ω_x, ω_y, ω_z] به رادیان بر ثانیه
    output = compute_euler_data(angular_velocity_vector)
    print(output)

    #سوال اول

    import math


    def compute_angular_velocities(V, bank_angle_deg):
        """
        این تابع سرعت خطی هواپیما (V به متر بر ثانیه) و زاویه بانک (به درجه) را دریافت می‌کند
        و یک دیکشنری با دو کلید زیر برمی‌گرداند:
          - "Angular_velocity_in_internal_frame": سرعت زاویه‌ای در دستگاه اینرسی به صورت لیست [0, 0, ω]
          - "Angular_velocity_in_body_frame": سرعت زاویه‌ای در دستگاه بدنه به صورت لیست [p, q, r]
            که در پرواز دور موزون (با زاویه پیچ صفر):
              p = 0
              q = sin(bank_angle) * ω
              r = cos(bank_angle) * ω
        """
        g = 9.81  # شتاب گرانش (m/s^2)

        # تبدیل زاویه بانک از درجه به رادیان
        bank_angle_rad = math.radians(bank_angle_deg)

        # محاسبه شعاع چرخش
        R = V ** 2 / (g * math.tan(bank_angle_rad))

        # محاسبه سرعت زاویه‌ای
        omega = V / R

        # سرعت زاویه‌ای در دستگاه اینرسی: فرض می‌کنیم چرخش تنها حول محور عمودی (z) رخ می‌دهد
        angular_velocity_in_internal_frame = [0.0, 0.0, omega]

        # سرعت زاویه‌ای در دستگاه بدنه:
        # p = 0, q = sin(bank_angle)*omega, r = cos(bank_angle)*omega
        p = 0.0
        q = math.sin(bank_angle_rad) * omega
        r = math.cos(bank_angle_rad) * omega
        angular_velocity_in_body_frame = [p, q, r]

        return {
            "Angular_velocity_in_internal_frame": angular_velocity_in_internal_frame,
            "Angular_velocity_in_body_frame": angular_velocity_in_body_frame
        }


    # مثال استفاده:
    V = 250  # سرعت خطی هواپیما به متر بر ثانیه
    bank_angle = 60  # زاویه بانک به درجه

    result = compute_angular_velocities(V, bank_angle)
    print(result)

