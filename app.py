import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & TITLE
# ----------------------------------------------------
st.set_page_config(
    page_title="Inverted Paraboloid Droplet Simulator",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Inverted Parabolic Disk Droplet Simulator")
st.caption("โปรแกรมจำลองและทำนายวิถีการกระจายตัวของหยดน้ำบนพื้นผิวจานหมุนทรงพาราโบลาคว่ำ (คำนวณด้วย RK4)")
st.markdown("---")

# ----------------------------------------------------
# 2. SIDEBAR - ALL 13 INPUT PARAMETERS
# ----------------------------------------------------
st.sidebar.header("⚙️ ปรับแต่งตัวแปร (Input Parameters)")

# Group 1: Disk Geometry & Motion
with st.sidebar.expander("📌 1. รูปทรงจานและการหมุน", expanded=True):
    omega_rpm = st.number_input("ความเร็วรอบการหมุน (omega_rpm) [RPM]", value=1031.18, step=10.0, format="%.2f")
    a = st.number_input("สัมประสิทธิ์ความโค้งจาน (a) [m⁻¹]", value=0.318, format="%.4f")
    H = st.number_input("ความสูงยอดจานจากพื้น (H) [m]", value=0.52, format="%.3f")
    R_max = st.number_input("รัศมีขอบจานสูงสุด (R_max) [m]", value=0.078, format="%.4f")

# Group 2: Liquid & Droplet Properties
with st.sidebar.expander("💧 2. คุณสมบัติของไหลและหยดน้ำ", expanded=True):
    r_d = st.number_input("รัศมีหยดน้ำ (r_d) [m]", value=0.00025, format="%.6f")
    h_film = st.number_input("ความหนาฟิล์มน้ำ (h_film) [m]", value=0.0001, format="%.6f")
    mu_s = st.number_input("สัมประสิทธิ์แรงยึดติด (mu_s)", value=8.22, format="%.2f")
    rho_w = st.number_input("ความหนาแน่นน้ำ (rho_w) [kg/m³]", value=1000.0, format="%.1f")
    mu_w = st.number_input("ความหนืดเชิงพลศาสตร์น้ำ (mu_w) [Pa·s]", value=0.001002, format="%.6f")
    gamma = st.number_input("แรงตึงผิวน้ำ (gamma) [N/m]", value=0.072, format="%.4f")

# Group 3: Environment & Physics Constants
with st.sidebar.expander("🌤️ 3. สิ่งแวดล้อมและแรงต้านอากาศ", expanded=False):
    Cd = st.number_input("สัมประสิทธิ์แรงต้านอากาศ (Cd)", value=0.47, format="%.2f")
    rho_air = st.number_input("ความหนาแน่นอากาศ (rho_air) [kg/m³]", value=1.20, format="%.2f")
    g = st.number_input("ความเร่งโน้มถ่วง (g) [m/s²]", value=9.81, format="%.2f")

# ----------------------------------------------------
# 3. PHYSICS & SIMULATION ENGINE
# ----------------------------------------------------
# 3.1 Angular Velocity and Mass Calculations
Omega = omega_rpm * (2.0 * math.pi / 60.0)
m_droplet = (4.0 / 3.0) * math.pi * (r_d**3) * rho_w
A_droplet = math.pi * (r_d**2)
B_drag = 0.5 * Cd * rho_air * A_droplet / m_droplet

# 3.2 Detachment Loop Verification
steps = 1000
r_detach = R_max
u_s_detach = 0.0

for i in range(1, steps + 1):
    r_curr = 0.001 + (R_max - 0.001) * (i / steps)
    cos_alpha = 1.0 / math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2))
    sin_alpha = 2.0 * a * r_curr / math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2))
    
    # Sliding velocity along surface
    u_s = (rho_w * r_curr * (h_film**2) * (Omega**2 + 2.0 * a * g)) / (3.0 * mu_w * math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2)))
    u_s = max(0.0, u_s)
    
    # Rotational Weber Number Check (We_r >= 1.0)
    We_r = (rho_w * (Omega**2) * r_curr * (h_film**2)) / gamma
    
    # Normal Force Breakdown Terms
    F_adhesion = mu_s * 2.0 * math.pi * r_d * gamma
    lift_term = (2.0 * a * (Omega**2) * (r_curr**2)) / math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2)) + \
                (2.0 * a * (u_s**2)) / ((1.0 + 4.0 * (a**2) * (r_curr**2))**1.5)
    gravity_normal = g / math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2))
    
    # Detachment Condition Check
    if We_r >= 1.0 or lift_term >= (gravity_normal + F_adhesion / m_droplet):
        r_detach = r_curr
        u_s_detach = u_s
        break

if r_detach == R_max and u_s_detach == 0.0:
    u_s_detach = (rho_w * R_max * (h_film**2) * (Omega**2 + 2.0 * a * g)) / (3.0 * mu_w * math.sqrt(1.0 + 4.0 * (a**2) * (R_max**2)))

# 3.3 Initial State Vectors at Release
z0 = H - a * (r_detach**2)
alpha_detach = math.atan(2.0 * a * r_detach)

vx0 = u_s_detach * math.cos(alpha_detach)
vy0 = Omega * r_detach
vz0 = -u_s_detach * math.sin(alpha_detach)
v_detach_mag = math.sqrt(vx0**2 + vy0**2 + vz0**2)

# 3.4 3D Trajectory Calculation using RK4
dt = 0.0005
t = 0.0
x, y, z = r_detach, 0.0, z0
vx, vy, vz = vx0, vy0, vz0

path_x, path_y, path_z = [x], [y], [z]

while z > 0 and t < 5.0:
    def derivatives(v_x, v_y, v_z):
        v_mag = math.sqrt(v_x**2 + v_y**2 + v_z**2)
        ax = -B_drag * v_mag * v_x
        ay = -B_drag * v_mag * v_y
        az = -g - B_drag * v_mag * v_z
        return ax, ay, az

    # RK4 Numerical Integration
    ax1, ay1, az1 = derivatives(vx, vy, vz)
    ax2, ay2, az2 = derivatives(vx + 0.5*dt*ax1, vy + 0.5*dt*ay1, vz + 0.5*dt*az1)
    ax3, ay3, az3 = derivatives(vx + 0.5*dt*ax2, vy + 0.5*dt*ay2, vz + 0.5*dt*az2)
    ax4, ay4, az4 = derivatives(vx + dt*ax3, vy + dt*ay3, vz + dt*az3)

    vx += (dt / 6.0) * (ax1 + 2*ax2 + 2*ax3 + ax4)
    vy += (dt / 6.0) * (ay1 + 2*ay2 + 2*ay3 + ay4)
    vz += (dt / 6.0) * (az1 + 2*az2 + 2*az3 + az4)

    x += vx * dt
    y += vy * dt
    z += vz * dt
    t += dt

    path_x.append(x)
    path_y.append(y)
    path_z.append(max(0.0, z))

R_splash_m = math.sqrt(x**2 + y**2)
R_splash_cm = R_splash_m * 100.0

# ----------------------------------------------------
# 4. KPI METRICS CARDS
# ----------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("รัศมีสลัดหลุด (r_detach)", f"{r_detach*100:.2f} cm")
col2.metric("ความเร็วหลุดออก (v_detach)", f"{v_detach_mag:.2f} m/s")
col3.metric("ระยะตกกระทบพื้น (Theory)", f"{R_splash_cm:.2f} cm")
col4.metric("เวลาลอยในอากาศ (t_fly)", f"{t:.3f} s")

st.markdown("---")

# ----------------------------------------------------
# 5. VISUALIZATION & COMPARISON DASHBOARD
# ----------------------------------------------------
col_graph, col_info = st.columns([2, 1])

with col_graph:
    st.subheader("🌐 แบบจำลองวิถี 3 มิติ (Interactive 3D Trajectory)")
    
    # Generate 3D Paraboloid Geometry Mesh
    r_mesh = np.linspace(0, R_max, 35)
    theta_mesh = np.linspace(0, 2 * np.pi, 35)
    R_grid, THETA_grid = np.meshgrid(r_mesh, theta_mesh)
    X_dish = R_grid * np.cos(THETA_grid)
    Y_dish = R_grid * np.sin(THETA_grid)
    Z_dish = H - a * (R_grid**2)

    fig = go.Figure()

    # 3D Parabolic Surface
    fig.add_trace(go.Surface(
        x=X_dish, y=Y_dish, z=Z_dish,
        colorscale='Blues', opacity=0.75, name="Parabolic Dish", showscale=False
    ))

    # 3D Droplet Trajectory Curve
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines', line=dict(color='red', width=6), name='Droplet Trajectory'
    ))

    # Ground Impact Marker
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[0],
        mode='markers', marker=dict(size=8, color='gold', symbol='diamond'), name='Impact Point'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=550
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.subheader("📊 เปรียบเทียบผลการทดลองจริง")
    
    exp_data = {
        1031.18: 169.60,
        1321.79: 190.74,
        1441.77: 198.68
    }
    
    closest_rpm = min(exp_data.keys(), key=lambda k: abs(k - omega_rpm))
    exp_dist = exp_data[closest_rpm]
    err_percent = abs(R_splash_cm - exp_dist) / exp_dist * 100.0

    st.markdown(f"**ความเร็วรอบอ้างอิง:** `{closest_rpm:.2f} RPM`")
    st.write(f"• ทฤษฎี (Theory): **{R_splash_cm:.2f} cm**")
    st.write(f"• ทดลองจริง (Experiment): **{exp_dist:.2f} cm**")
    st.write(f"• ผลต่าง (Delta): **{abs(R_splash_cm - exp_dist):.2f} cm**")
    
    st.success(f"🎯 **ความคลาดเคลื่อน (% Error): {err_percent:.2f}%**")
    
    st.markdown("---")
    st.subheader("⚙️ สรุปค่าพารามิเตอร์ปัจจุบัน")
    st.json({
        "Omega (rad/s)": round(Omega, 2),
        "Mass (kg)": f"{m_droplet:.3e}",
        "Drag Parameter B (m⁻¹)": round(B_drag, 4),
        "Detachment Radius (m)": round(r_detach, 4)
    })