import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Inverted Paraboloid Droplet Theoretical Simulator",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 3D Parabolic Disk Droplet Dynamics Simulator")
st.caption("โปรแกรมจำลองวิถีการกระจายตัวของหยดน้ำแบบ 3 มิติเชิงทฤษฎี (RK4 & Coriolis Dynamics)")
st.markdown("---")

# ----------------------------------------------------
# 2. SIDEBAR - PARAMETERS & UNIT SWITCHER
# ----------------------------------------------------
st.sidebar.header("⚙️ ปรับแต่งตัวแปร (Input Parameters)")

# Unit Switcher Toggle
unit_choice = st.sidebar.radio(
    "📏 เลือกหน่วยการแสดงผลพิกัด (Display Unit)",
    ["เซนติเมตร (cm)", "เมตร (m)"],
    index=0
)

if unit_choice == "เซนติเมตร (cm)":
    u_scale = 100.0
    u_label = "cm"
else:
    u_scale = 1.0
    u_label = "m"

with st.sidebar.expander("📌 1. รูปทรงจานและการหมุน", expanded=True):
    omega_rpm = st.number_input("ความเร็วรอบการหมุน (omega_rpm) [RPM]", value=1031.18, step=10.0, format="%.2f")
    a = st.number_input("สัมประสิทธิ์ความโค้งจาน (a) [m⁻¹]", value=0.318, format="%.4f")
    H = st.number_input("ความสูงยอดจานจากพื้น (H) [m]", value=0.52, format="%.4f")
    R_max = st.number_input("รัศมีขอบจานสูงสุด (R_max) [m]", value=0.078, format="%.4f")

with st.sidebar.expander("💧 2. คุณสมบัติของไหลและหยดน้ำ", expanded=True):
    r_d = st.number_input("รัศมีหยดน้ำ (r_d) [m]", value=0.00025, format="%.6f")
    h_film = st.number_input("ความหนาฟิล์มน้ำ (h_film) [m]", value=0.0001, format="%.6f")
    mu_s = st.number_input("สัมประสิทธิ์แรงยึดติด (mu_s)", value=8.22, format="%.2f")
    rho_w = st.number_input("ความหนาแน่นน้ำ (rho_w) [kg/m³]", value=1000.0, format="%.1f")
    mu_w = st.number_input("ความหนืดเชิงพลศาสตร์น้ำ (mu_w) [Pa·s]", value=0.001002, format="%.6f")
    gamma = st.number_input("แรงตึงผิวน้ำ (gamma) [N/m]", value=0.072, format="%.4f")

with st.sidebar.expander("🌤️ 3. สิ่งแวดล้อมและแรงต้านอากาศ", expanded=False):
    Cd = st.number_input("สัมประสิทธิ์แรงต้านอากาศ (Cd)", value=0.47, format="%.2f")
    rho_air = st.number_input("ความหนาแน่นอากาศ (rho_air) [kg/m³]", value=1.20, format="%.2f")
    g = st.number_input("ความเร่งโน้มถ่วง (g) [m/s²]", value=9.81, format="%.2f")

# ----------------------------------------------------
# 3. PHYSICS & SIMULATION CALCULATIONS
# ----------------------------------------------------
Omega = omega_rpm * (2.0 * math.pi / 60.0)
m_droplet = (4.0 / 3.0) * math.pi * (r_d**3) * rho_w
A_droplet = math.pi * (r_d**2)
B_drag = 0.5 * Cd * rho_air * A_droplet / m_droplet

# 3.1 Detachment Radius Search
steps = 1000
r_detach = R_max
u_s_detach = 0.0

for i in range(1, steps + 1):
    r_curr = 0.001 + (R_max - 0.001) * (i / steps)
    cos_alpha = 1.0 / math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2))
    
    u_s = (rho_w * r_curr * (h_film**2) * (Omega**2 + 2.0 * a * g)) / (3.0 * mu_w * math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2)))
    u_s = max(0.0, u_s)
    
    We_r = (rho_w * (Omega**2) * r_curr * (h_film**2)) / gamma
    F_adhesion = mu_s * 2.0 * math.pi * r_d * gamma
    lift_term = (2.0 * a * (Omega**2) * (r_curr**2)) / math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2)) + \
                (2.0 * a * (u_s**2)) / ((1.0 + 4.0 * (a**2) * (r_curr**2))**1.5)
    gravity_normal = g / math.sqrt(1.0 + 4.0 * (a**2) * (r_curr**2))
    
    if We_r >= 1.0 or lift_term >= (gravity_normal + F_adhesion / m_droplet):
        r_detach = r_curr
        u_s_detach = u_s
        break

if r_detach == R_max and u_s_detach == 0.0:
    u_s_detach = (rho_w * R_max * (h_film**2) * (Omega**2 + 2.0 * a * g)) / (3.0 * mu_w * math.sqrt(1.0 + 4.0 * (a**2) * (R_max**2)))

z0 = H - a * (r_detach**2)
alpha_detach = math.atan(2.0 * a * r_detach)
v_radial_0 = u_s_detach * math.cos(alpha_detach)
v_tan_0 = Omega * r_detach
v_vert_0 = -u_s_detach * math.sin(alpha_detach)
v_detach_mag = math.sqrt(v_radial_0**2 + v_tan_0**2 + v_vert_0**2)

# 3.2 On-Disk Relative Spiral Angle & Velocity Integration
dish_r_pts = np.linspace(0.001, r_detach, 120)
dish_theta_rel = [0.0]
v_disk_pts = []

# Calculate local velocity at r=0.001
u_s_start = (rho_w * dish_r_pts[0] * (h_film**2) * (Omega**2 + 2.0 * a * g)) / (3.0 * mu_w * math.sqrt(1.0 + 4.0 * (a**2) * (dish_r_pts[0]**2)))
v_disk_pts.append(math.sqrt(u_s_start**2 + (Omega * dish_r_pts[0])**2))

current_theta = 0.0
for idx in range(len(dish_r_pts) - 1):
    r_c = dish_r_pts[idx]
    dr = dish_r_pts[idx+1] - dish_r_pts[idx]
    
    u_s_c = (rho_w * r_c * (h_film**2) * (Omega**2 + 2.0 * a * g)) / (3.0 * mu_w * math.sqrt(1.0 + 4.0 * (a**2) * (r_c**2)))
    u_s_c = max(1e-5, u_s_c)
    
    # Calculate local total velocity v(r)
    v_tan_c = Omega * r_c
    v_total_c = math.sqrt(u_s_c**2 + v_tan_c**2)
    v_disk_pts.append(v_total_c)
    
    # Differential angle step dt and dtheta due to rotation/coriolis lag
    cos_a_c = 1.0 / math.sqrt(1.0 + 4.0 * (a**2) * (r_c**2))
    dt_step = dr / (u_s_c * cos_a_c)
    dtheta = (Omega) * dt_step
    current_theta += dtheta
    dish_theta_rel.append(current_theta)

dish_theta_rel = np.array(dish_theta_rel)
theta_end = dish_theta_rel[-1]  # Final relative angle at r_detach

# 3.3 Airborne Stream Simulation Function (RK4)
def simulate_airborne_stream(phi_angle):
    vx0 = v_radial_0 * math.cos(phi_angle) - v_tan_0 * math.sin(phi_angle)
    vy0 = v_radial_0 * math.sin(phi_angle) + v_tan_0 * math.cos(phi_angle)
    vz0 = v_vert_0

    dt = 0.0005
    t = 0.0
    x = r_detach * math.cos(phi_angle)
    y = r_detach * math.sin(phi_angle)
    z = z0
    vx, vy, vz = vx0, vy0, vz0

    px, py, pz, pv, pt = [x], [y], [z], [math.sqrt(vx0**2 + vy0**2 + vz0**2)], [0.0]

    while z > 0 and t < 5.0:
        def derivatives(v_x, v_y, v_z):
            v_mag = math.sqrt(v_x**2 + v_y**2 + v_z**2)
            return -B_drag * v_mag * v_x, -B_drag * v_mag * v_y, -g - B_drag * v_mag * v_z

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

        v_inst = math.sqrt(vx**2 + vy**2 + vz**2)
        px.append(x)
        py.append(y)
        pz.append(max(0.0, z))
        pv.append(v_inst)
        pt.append(t)

    return px, py, pz, pv, pt, math.sqrt(x**2 + y**2)

# ----------------------------------------------------
# 4. KPI METRICS
# ----------------------------------------------------
first_px, first_py, first_pz, first_pv, first_pt, R_splash_m = simulate_airborne_stream(0.0)

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"รัศมีสลัดหลุด (r_detach)", f"{r_detach * u_scale:.2f} {u_label}")
col2.metric("ความเร็วหลุดออก (v_detach)", f"{v_detach_mag:.2f} m/s")
col3.metric(f"ระยะตกกระทบพื้น (R_splash)", f"{R_splash_m * u_scale:.2f} {u_label}")
col4.metric("เวลาลอยในอากาศ (t_fly)", f"{first_pt[-1]:.3f} s")

st.markdown("---")

# ----------------------------------------------------
# 5. 3D VISUALIZATION (PLOTLY)
# ----------------------------------------------------
st.subheader(f"🌐 แบบจำลอง 3 มิติ: การกระจายตัว 8 ทิศทาง (สเกลพิกัด: {u_label})")

fig = go.Figure()

# 5.1 Paraboloid Surface Mesh (Scaled)
r_mesh = np.linspace(0, R_max, 45)
theta_mesh = np.linspace(0, 2 * np.pi, 70)
R_grid, THETA_grid = np.meshgrid(r_mesh, theta_mesh)
X_dish = (R_grid * np.cos(THETA_grid)) * u_scale
Y_dish = (R_grid * np.sin(THETA_grid)) * u_scale
Z_dish = (H - a * (R_grid**2)) * u_scale

fig.add_trace(go.Surface(
    x=X_dish, y=Y_dish, z=Z_dish,
    colorscale='Ice', opacity=0.70, name="3D Parabolic Dish", showscale=False
))

# 5.2 Generate 8 On-Disk Paths + 8 Airborne Streams
num_streams = 8
phi_angles = np.linspace(0, 2 * np.pi, num_streams, endpoint=False)
impact_x, impact_y = [], []

for idx, phi in enumerate(phi_angles):
    # A. Continuous On-Disk Spiral Path (Ends exactly at detachment angle phi)
    angle_profile = phi - (theta_end - dish_theta_rel)
    disk_x = (dish_r_pts * np.cos(angle_profile)) * u_scale
    disk_y = (dish_r_pts * np.sin(angle_profile)) * u_scale
    disk_z = (H - a * (dish_r_pts**2)) * u_scale
    disk_r_disp = dish_r_pts * u_scale
    
    customdata_disk = np.stack((v_disk_pts, disk_r_disp), axis=-1)
    
    fig.add_trace(go.Scatter3d(
        x=disk_x, y=disk_y, z=disk_z,
        mode='lines',
        line=dict(color='navy', width=5),
        customdata=customdata_disk,
        hovertemplate="<b>เส้นทางการไหลบนผิวจาน (Coriolis)</b><br>" +
                      f"พิกัด X: %{{x:.2f}} {u_label}<br>" +
                      f"พิกัด Y: %{{y:.2f}} {u_label}<br>" +
                      f"ความสูง Z: %{{z:.2f}} {u_label}<br>" +
                      "<b>ความเร็ว ณ รัศมีนี้ (v): %{customdata[0]:.2f} m/s</b><br>" +
                      f"รัศมี (r): %{{customdata[1]:.2f}} {u_label}<extra></extra>",
        name='เส้นทางการไหลบนผิวจาน (8 เส้น)' if idx == 0 else "",
        showlegend=(idx == 0)
    ))

    # B. Airborne Trajectory
    px, py, pz, pv, pt, R_sp = simulate_airborne_stream(phi)
    px_disp = np.array(px) * u_scale
    py_disp = np.array(py) * u_scale
    pz_disp = np.array(pz) * u_scale
    
    impact_x.append(px_disp[-1])
    impact_y.append(py_disp[-1])
    
    customdata_air = np.stack((pv, pt), axis=-1)
    
    fig.add_trace(go.Scatter3d(
        x=px_disp, y=py_disp, z=pz_disp,
        mode='lines',
        line=dict(color='crimson', width=6),
        customdata=customdata_air,
        hovertemplate="<b>วิถีหยดน้ำในอากาศ</b><br>" +
                      f"พิกัด X: %{{x:.2f}} {u_label}<br>" +
                      f"พิกัด Y: %{{y:.2f}} {u_label}<br>" +
                      f"ความสูง Z: %{{z:.2f}} {u_label}<br>" +
                      "<b>ความเร็วขณะใดๆ (v): %{customdata[0]:.2f} m/s</b><br>" +
                      "เวลา (t): %{customdata[1]:.3f} s<extra></extra>",
        name='วิถีหยดน้ำในอากาศ (8 เส้น)' if idx == 0 else "",
        showlegend=(idx == 0)
    ))

# 5.3 Splash Ring & Impact Markers
ring_theta = np.linspace(0, 2 * np.pi, 120)
ring_x = (R_splash_m * np.cos(ring_theta)) * u_scale
ring_y = (R_splash_m * np.sin(ring_theta)) * u_scale

fig.add_trace(go.Scatter3d(
    x=ring_x, y=ring_y, z=np.zeros_like(ring_x),
    mode='lines', line=dict(color='gold', width=4, dash='dash'),
    name='วงแหวนรัศมีตกกระทบพื้น',
    hoverinfo='skip'
))

fig.add_trace(go.Scatter3d(
    x=impact_x, y=impact_y, z=np.zeros(num_streams),
    mode='markers', marker=dict(size=7, color='gold', symbol='diamond'),
    name='จุดตกกระทบพื้น (8 จุด)',
    hoverinfo='skip'
))

# 5.4 3D Scene Ratio & Camera Configuration
fig.update_layout(
    scene=dict(
        xaxis_title=f'X ({u_label})',
        yaxis_title=f'Y ({u_label})',
        zaxis_title=f'ความสูง Z ({u_label})',
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.6),
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.1))
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    height=680
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 6. MATHEMATICAL SUMMARY TABLE
# ----------------------------------------------------
st.markdown("---")
st.subheader("📋 สรุปพารามิเตอร์และผลลัพธ์ทางทฤษฎี")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown("**สภาวะ ณ จุดสลัดหลุด (Detachment Conditions):**")
    st.write(f"• รัศมีจุดสลัดหลุด ($r_{{detach}}$): `{r_detach * u_scale:.2f} {u_label}` (`{r_detach:.4f} m`) ")
    st.write(f"• ความสูงจุดสลัดหลุด ($z_0$): `{z0 * u_scale:.2f} {u_label}` (`{z0:.4f} m`) ")
    st.write(f"• ความเร็วสไลด์ตามแนวผิว ($u_s$): `{u_s_detach:.3f} m/s`")
    st.write(f"• ความเร็วตามแนวเส้นสัมผัส ($v_\\theta$): `{v_tan_0:.3f} m/s`")

with summary_col2:
    st.markdown("**ผลลัพธ์การเคลื่อนที่ในอากาศ (3D Trajectory Output):**")
    st.write(f"• ความเร็วหลุดออกสัมบูรณ์ ($v_{{detach}}$): `{v_detach_mag:.3f} m/s`")
    st.write(f"• ระยะตกกระทบพื้นทางทฤษฎี ($R_{{splash}}$): `{R_splash_m * u_scale:.2f} {u_label}` (`{R_splash_m:.3f} m`) ")
    st.write(f"• เวลาที่ใช้ในการลอย ($t_{{fly}}$): `{first_pt[-1]:.4f} s`")
    st.write(f"• พารามิเตอร์แรงต้านอากาศ ($B_{{drag}}$): `{B_drag:.4f} m⁻¹`")