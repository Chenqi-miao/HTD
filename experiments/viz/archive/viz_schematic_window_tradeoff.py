import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

# 示意图:锚点取项目真实趋势(s8/s15/s31)
win   = np.array([8, 15, 31])
mpjpe = np.array([9.0, 10.0, 11.6])   # 位置误差(mm),随窗口升高
corr  = np.array([0.32, 0.38, 0.43])  # 运动相关性,随窗口提升

fig, ax1 = plt.subplots(figsize=(9.5, 6.2))
ax2 = ax1.twinx()

l1, = ax1.plot(win, mpjpe, 'o-', color='#d62728', lw=3.0, ms=10,
               label='位置误差 MPJPE (越低越好)')
ax1.set_xticks(win); ax1.set_xticklabels(['s8', 's15', 's31'], fontsize=13)
ax1.set_xlabel('窗口长度 (帧)', fontsize=13)
ax1.set_ylabel('位置误差 MPJPE (mm)  ↑', fontsize=13, color='#d62728')
ax1.tick_params(axis='y', labelcolor='#d62728')
ax1.set_ylim(8.2, 12.6)
for x, y in zip(win, mpjpe):
    ax1.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                 xytext=(0, 9), ha='center', fontsize=11, color='#d62728')

l2, = ax2.plot(win, corr, 's-', color='#1f77b4', lw=3.0, ms=10,
               label='运动相关性 corr (越高越好)')
ax2.set_ylabel('运动相关性 corr  ↑', fontsize=13, color='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#1f77b4')
ax2.set_ylim(0.28, 0.50)
for x, y in zip(win, corr):
    ax2.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                 xytext=(0, -18), ha='center', fontsize=11, color='#1f77b4')

# 两端含义注释
ax1.annotate('短窗: 位置准 / 运动差', xy=(8.5, 9.1), xytext=(8.8, 8.55),
             fontsize=12, color='#555555')
ax1.annotate('长窗: 运动准 / 位置差', xy=(31, 11.6), xytext=(17.5, 12.2),
             fontsize=12, color='#555555')

# 核心结论
ax1.text(0.5, 0.08, '单个窗口无法两全  →  需将空间/时间修正解耦',
         transform=ax1.transAxes, ha='center', fontsize=13,
         bbox=dict(boxstyle='round,pad=0.45', fc='#fff8e1', ec='#f0c040', lw=1.5))

lines = [l1, l2]
ax1.legend(handles=lines, loc='center left', bbox_to_anchor=(0.03, 0.88), fontsize=12, frameon=False)
ax1.set_title('窗口长度 vs 位置精度 / 运动保真的权衡', fontsize=14, pad=12)
ax1.grid(alpha=0.25, linestyle='--')
fig.tight_layout()
out = '/mnt/d/WORKSPACE/NIRC考核/汇报总结/图片和附件/schematic_window_tradeoff.png'
fig.savefig(out, dpi=180)
print('saved', out)
