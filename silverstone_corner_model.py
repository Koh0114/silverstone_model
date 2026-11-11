"""
다운포스를 고려한 F1 코너 최고속도 모델 — 2025 영국 GP(실버스톤) 예선 데이터로 검증

모델:  m·v²/r = μ·(m·g + ½·ρ·C_L·A·v²)
직선화: v²/r = μ·g + (μ·ρ·C_L·A / 2m)·v²      →  y = a + b·x
        a = μ·g,  b = μ·ρ·C_L·A / (2m),  1/b = 임계 반경

[데이터]
  이 파일 옆에 'data' 폴더(CSV)가 있으면 그걸 씀 → 인터넷 필요 없음
  없으면 FastF1으로 F1 서버에서 받아서 data 폴더에 CSV로 저장 (서버가 막히는 경우가 있음)

설치:  pip install -r requirements.txt
실행:  python silverstone_corner_model.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ───────────────────────── 내가 정한 값 (심화 1번) ─────────────────────────
YEAR, GP, SESSION = 2025, 'British Grand Prix', 'Q'   # 예선: 혼자 달림, 새 타이어, 연료 적음
MASS = 800.0          # kg, 2025 최저 중량(드라이버 포함). 예선 연료는 적어서 무시
G = 9.81
MIN_DROP_KMH = 8      # 속도가 이만큼 이상 떨어졌다 올라가야 '코너'로 인정
FIT_HALF_M = 35       # 정점 앞뒤 몇 m의 좌표로 원을 맞출지 (반경 측정)
THROTTLE_MAX = 98     # 정점에서 스로틀이 이 이상이면 풀스로틀 → 그립 한계 아님 → 제외
FLAT_THROTTLE = 98    # 코너 구간 최소 스로틀이 이 이상이면 '풀스로틀 코너'로 분류

NAMES = {1: 'Abbey', 2: 'Farm', 3: 'Village', 4: 'Loop', 6: 'Brooklands', 7: 'Luffield',
         8: 'Woodcote', 9: 'Copse', 10: 'Maggotts', 13: 'Becketts', 15: 'Stowe',
         16: 'Vale', 17: 'Club'}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')          # 결과(그래프·CSV) 저장 폴더
DATA = os.path.join(HERE, 'data')        # 오프라인 데이터 폴더
os.makedirs(OUT, exist_ok=True)


def fit_circle(x, y):
    """최소제곱 원 맞춤: x²+y² = 2cx·x + 2cy·y + c  →  반경 = √(c + cx² + cy²)"""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.sqrt(c + cx ** 2 + cy ** 2)


def air_density(T_c, p_hpa, rh):
    """습도까지 고려한 공기 밀도 (건조공기 + 수증기 부분압)"""
    T = T_c + 273.15
    p_sat = 6.1078 * 10 ** (7.5 * T_c / (T_c + 237.3))   # 포화수증기압 (hPa, Tetens 식)
    p_v = rh / 100 * p_sat
    p_d = p_hpa - p_v
    return (p_d * 100) / (287.05 * T) + (p_v * 100) / (461.5 * T)


def load_from_fastf1():
    """F1 서버에서 받아서 data 폴더에 CSV로 저장까지 함"""
    import fastf1
    from fastf1.exceptions import DataNotLoadedError
    cache = os.path.join(os.path.expanduser('~'), 'f1_cache')   # OneDrive 밖에 캐시
    os.makedirs(cache, exist_ok=True)
    fastf1.Cache.enable_cache(cache)
    s = fastf1.get_session(YEAR, GP, SESSION)
    s.load(messages=False)
    try:
        corners = s.get_circuit_info().corners[['Number', 'Distance']]
        laps_all = s.laps
    except DataNotLoadedError:
        raise SystemExit('\n[데이터 불러오기 실패] F1 서버가 요청을 막았습니다.\n'
                         '→ 다른 네트워크에서 다시 시도하거나, 같은 형식의 CSV가 든 data 폴더를 이 파일 옆에 두고 실행하세요.')
    laps, car, pos = [], [], []
    for drv in laps_all['Driver'].unique():
        lap = laps_all.pick_drivers(drv).pick_fastest()
        if lap is None or pd.isna(lap['LapTime']):
            continue
        c = lap.get_car_data().add_distance()
        p = lap.get_pos_data()
        laps.append(dict(driver=drv, team=lap['Team'], laptime_s=lap['LapTime'].total_seconds()))
        car.append(pd.DataFrame(dict(driver=drv, t=c['SessionTime'].dt.total_seconds(),
                                     Distance=c['Distance'], Speed=c['Speed'],
                                     Throttle=c['Throttle'], Brake=c['Brake'].astype(int))))
        pos.append(pd.DataFrame(dict(driver=drv, t=p['SessionTime'].dt.total_seconds(),
                                     X=p['X'], Y=p['Y'])))
    d = dict(laps=pd.DataFrame(laps), car=pd.concat(car), pos=pd.concat(pos), corners=corners,
             weather=s.weather_data[['AirTemp', 'Pressure', 'Humidity', 'Rainfall']])
    os.makedirs(DATA, exist_ok=True)
    for k, v in d.items():
        v.to_csv(os.path.join(DATA, f'{k}.csv'), index=False)
    return d


def load_from_csv():
    return {k: pd.read_csv(os.path.join(DATA, f'{k}.csv'))
            for k in ['laps', 'car', 'pos', 'corners', 'weather']}


# ───────────────────────── 1. 데이터 불러오기 ─────────────────────────
if os.path.exists(os.path.join(DATA, 'car.csv')):
    print('data 폴더의 CSV 사용 (오프라인)')
    D = load_from_csv()
else:
    print('F1 서버에서 데이터 받는 중...')
    D = load_from_fastf1()
corners, w = D['corners'], D['weather']
assert not w['Rainfall'].astype(bool).any(), '비가 온 세션 — 드라이 조건이 아님'
rho = air_density(w['AirTemp'].mean(), w['Pressure'].mean(), w['Humidity'].mean())
print(f'공기 밀도 ρ = {rho:.4f} kg/m³  (기온 {w.AirTemp.mean():.1f}°C, 기압 {w.Pressure.mean():.1f} hPa)')

# ───────────────────────── 2. 드라이버별 최고 랩에서 코너 정점 뽑기 ─────────────────────────
apex_rows, flat_rows = [], []
for _, L in D['laps'].iterrows():
    drv = L['driver']
    car = D['car'][D['car'].driver == drv]           # 속도·스로틀·브레이크 (약 4Hz)
    pos = D['pos'][D['pos'].driver == drv]           # 좌표 X, Y (단위 1/10 m)
    dist = car['Distance'].values
    v = car['Speed'].values / 3.6                    # m/s
    thr = car['Throttle'].values
    brk = car['Brake'].values.astype(bool)
    # 좌표 데이터에 주행거리 붙이기 (시간 기준 보간)
    d_pos = np.interp(pos['t'].values, car['t'].values, dist)
    X, Y = pos['X'].values / 10, pos['Y'].values / 10

    # (a) 속도 최저점 = 코너 정점
    peaks, _ = find_peaks(-car['Speed'].values, prominence=MIN_DROP_KMH)
    for i in peaks:
        near = np.abs(d_pos - dist[i]) < FIT_HALF_M
        if near.sum() < 4:
            continue
        turn = int(corners.loc[(corners['Distance'] - dist[i]).abs().idxmin(), 'Number'])
        apex_rows.append(dict(driver=drv, team=L['team'], turn=turn, v=v[i],
                              r=fit_circle(X[near], Y[near]), throttle=thr[i], brake=brk[i]))

    # (b) 풀스로틀 코너 (모델 성립 범위 확인용)
    for _, c in corners.iterrows():
        seg = np.abs(dist - c['Distance']) < 100
        near = np.abs(d_pos - c['Distance']) < 60
        if seg.sum() and thr[seg].min() >= FLAT_THROTTLE and near.sum() >= 4:
            flat_rows.append(dict(driver=drv, turn=int(c['Number']),
                                  v=v[seg].min(), r=fit_circle(X[near], Y[near])))

raw = pd.DataFrame(apex_rows)
# ───────────────────────── 3. 모델 가정이 성립하는 점만 남기기 ─────────────────────────
df = raw[(raw.throttle < THROTTLE_MAX) & (~raw.brake)].copy()
print(f'정점 {len(raw)}개 중 {len(df)}개 사용 (풀스로틀·브레이크 중인 점 제외)')
df['x'] = df.v ** 2             # v²
df['y'] = df.v ** 2 / df.r      # 횡가속도 v²/r
df.to_csv(f'{OUT}/apex_points.csv', index=False)

# ───────────────────────── 4. 직선 맞춤 → μ, C_L·A 역산 ─────────────────────────
b, a = np.polyfit(df.x, df.y, 1)
mu = a / G
CLA = 2 * MASS * b / (mu * rho)
r_crit = 1 / b
r2 = np.corrcoef(df.x, df.y)[0, 1] ** 2
print(f'\n절편 a = {a:.2f} m/s²   기울기 b = {b:.5f} 1/m   R² = {r2:.3f}')
print(f'→ μ = {mu:.2f}   C_L·A = {CLA:.2f} m²   임계 반경 = {r_crit:.0f} m')
print(f'→ 300 km/h에서 다운포스 = {0.5 * rho * CLA * (300 / 3.6) ** 2 / (MASS * G):.1f} × 차 무게')


def v_model(r):
    den = 1 - mu * rho * CLA * r / (2 * MASS)
    return np.where(den > 0, np.sqrt(mu * G * r / np.clip(den, 1e-9, None)), np.inf)


# ───────────────────────── 5. 코너별 예측 vs 실제 ─────────────────────────
per = df.groupby('turn').agg(n=('v', 'size'), r=('r', 'median'), v_real=('v', 'median'))
per['v_pred'] = v_model(per.r.values)
per['v_nodf'] = np.sqrt(mu * G * per.r)                  # 다운포스 없을 때
per['err_%'] = (per.v_pred / per.v_real - 1) * 100
per['name'] = [NAMES.get(t, '') for t in per.index]
show = per.copy()
for col in ['v_real', 'v_pred', 'v_nodf']:
    show[col] = show[col] * 3.6
print('\n코너별 비교 (km/h)')
print(show[['name', 'n', 'r', 'v_real', 'v_pred', 'v_nodf', 'err_%']].round(1).to_string())
print(f'평균 절대 오차 = {per["err_%"].abs().mean():.1f} %')
show.to_csv(f'{OUT}/corner_compare.csv')

# ───────────────────────── 6. 성립 범위 확인: 풀스로틀 코너의 반경 ─────────────────────────
flat = pd.DataFrame(flat_rows)
if len(flat):
    fl = flat.groupby('turn').agg(n=('v', 'size'), r=('r', 'median'), v=('v', 'median'))
    fl['name'] = [NAMES.get(t, '') for t in fl.index]
    fl['r > 임계반경?'] = fl.r > r_crit
    fl['v'] *= 3.6
    print(f'\n풀스로틀 코너 (모델 예측: 반경 > {r_crit:.0f} m 이면 그립 한계 없음)')
    print(fl.round(1).to_string())

# ───────────────────────── 7. 그래프 ─────────────────────────
BLUE, INK, MUTED, GRID = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False})

# 그래프 1: 직선화 그래프
fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.scatter(df.x, df.y / G, s=22, color=BLUE, alpha=0.45, edgecolor='white', lw=0.6,
           label='each driver, each corner')
xs = np.linspace(0, df.x.max() * 1.05, 50)
ax.plot(xs, (a + b * xs) / G, color=INK, lw=2, label=f'fit: μ={mu:.2f}, C_L·A={CLA:.1f} m²')
for t, row in df.groupby('turn')[['x', 'y']].median().iterrows():
    ax.annotate(NAMES.get(t, f'T{t}'), (row.x, row.y / G), xytext=(6, -10),
                textcoords='offset points', fontsize=8, color=MUTED)
ax.set_xlabel('v²  (m²/s²)'); ax.set_ylabel('lateral acceleration  v²/r  (g)')
ax.set_title(f'Silverstone 2025 Qualifying — grip-limited corners  (R² = {r2:.2f})',
             loc='left', fontsize=11, color=INK)
ax.legend(frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_linear_fit.png')

# 그래프 2: 반경에 따른 최고속도 곡선 + 실제 데이터
fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
rs = np.linspace(10, r_crit * 0.985, 300)
ax.plot(rs, v_model(rs) * 3.6, color=BLUE, lw=2, label='model with downforce')
ax.plot(rs, np.sqrt(mu * G * rs) * 3.6, color=MUTED, lw=2, ls='--', label='model without downforce')
ax.axvline(r_crit, color=MUTED, lw=1, ls=':')
ax.text(r_crit, 355, f'  critical radius {r_crit:.0f} m', color=MUTED, fontsize=8)
ax.scatter(per.r, per.v_real * 3.6, s=40, color=INK, zorder=3, label='real (grip-limited)')
if len(flat):
    ax.scatter(fl.r, fl.v, s=40, facecolor='white', edgecolor=INK, zorder=3,
               label='real (flat-out)')
ax.set_xlim(0, max(r_crit * 1.3, fl.r.max() * 1.05 if len(flat) else 0))
ax.set_ylim(0, 380)
ax.set_xlabel('corner radius r  (m)'); ax.set_ylabel('minimum corner speed  (km/h)')
ax.set_title('Predicted vs real corner speed', loc='left', fontsize=11, color=INK)
ax.legend(frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_speed_vs_radius.png')
print(f'\n그래프 저장: {OUT}/fig1_linear_fit.png, {OUT}/fig2_speed_vs_radius.png')

# ───────────────────────── 8. 공정한 검증: 절반으로 맞추고 나머지 절반 예측 ─────────────────────────
# 같은 데이터로 맞추고 같은 데이터로 오차를 재면 결과가 좋게 나오는 게 당연함.
# → 팀마다 드라이버 1명씩 A조, 1명씩 B조 (팀마다 차 성능이 달라서 두 조에 모든 팀이 들어가게)
#   팀 안에서는 이름 알파벳 순으로 나눔 (빠른 사람만 한쪽에 몰리지 않게)
#   A조로 μ, C_L·A를 구해서 B조 속도를 예측, 그다음 반대로도 (교차 검증)
groups = {}
for team, drv in D['laps'].groupby('team')['driver']:
    d = sorted(drv)
    groups.update({d[0]: 'A', d[1]: 'B'} if len(d) > 1 else {d[0]: 'A'})
df['group'] = df.driver.map(groups)
print('\n[공정한 검증]  A조:', ', '.join(sorted(k for k, g in groups.items() if g == 'A')))
print('               B조:', ', '.join(sorted(k for k, g in groups.items() if g == 'B')))


def fit_params(sub):
    b_, a_ = np.polyfit(sub.x, sub.y, 1)
    mu_ = a_ / G
    return mu_, 2 * MASS * b_ / (mu_ * rho)


def predict(r, mu_, cla_):
    den = 1 - mu_ * rho * cla_ * r / (2 * MASS)
    return np.where(den > 0, np.sqrt(mu_ * G * r / np.clip(den, 1e-9, None)), np.inf)


cv_rows = []
for train, test in [('A', 'B'), ('B', 'A')]:
    mu_t, cla_t = fit_params(df[df.group == train])
    te = df[df.group == test].groupby('turn').agg(r=('r', 'median'), v_real=('v', 'median'))
    te['v_pred'] = predict(te.r.values, mu_t, cla_t)
    te['err_%'] = (te.v_pred / te.v_real - 1) * 100
    print(f'{train}조로 맞춤 → μ = {mu_t:.2f}, C_L·A = {cla_t:.2f} m²  |  '
          f'{test}조 예측 평균 절대 오차 = {te["err_%"].abs().mean():.1f} %')
    for t, row in te.iterrows():
        cv_rows.append(dict(fold=f'{train}→{test}', turn=t, name=NAMES.get(t, f'T{t}'),
                            r=row.r, v_real=row.v_real * 3.6, v_pred=row.v_pred * 3.6,
                            err=row['err_%']))
cv = pd.DataFrame(cv_rows)
cv.round(1).to_csv(f'{OUT}/cross_validation.csv', index=False)
print(f'전체 데이터로 맞추고 같은 데이터로 잰 오차(비교용) = {per["err_%"].abs().mean():.1f} %')

# 그래프 3: 예측 vs 실제 (점이 대각선 위에 있을수록 정확)
ORANGE = '#eb6834'
fig, ax = plt.subplots(figsize=(5.6, 5.4), dpi=150)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
lim = [70, 260]
ax.plot(lim, lim, color=MUTED, lw=1, ls='--', label='perfect prediction')
for (fold, sub), col in zip(cv.groupby('fold', sort=False), [BLUE, ORANGE]):
    ax.scatter(sub.v_real, sub.v_pred, s=40, color=col, edgecolor='white', lw=1, zorder=3,
               label=f'fit {fold[0]} → predict {fold[-1]}  (error {sub.err.abs().mean():.1f}%)')
LABEL_OFFSET = {13: (8, -14), 16: (-34, 2), 4: (6, 4)}   # 글자 겹침 방지용 위치 조정
for t, sub in cv.groupby('turn'):
    ax.annotate(sub.name.iloc[0], (sub.v_real.mean(), sub.v_pred.max()),
                xytext=LABEL_OFFSET.get(t, (5, 4)), textcoords='offset points',
                fontsize=8, color=MUTED)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('real corner speed  (km/h)'); ax.set_ylabel('predicted corner speed  (km/h)')
ax.set_title('Cross-validation: fit on one half, predict the other', loc='left',
             fontsize=11, color=INK)
ax.legend(frameon=False, loc='upper left', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_cross_validation.png')
print(f'그래프 저장: {OUT}/fig3_cross_validation.png')
