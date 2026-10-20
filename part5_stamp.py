# Part 5 - STAMP Safety Analysis and Safety Governor
# Run this from inside your capstone folder

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────
# 1. STAMP Control Structure Diagram
# ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('STAMP Control Structure\nPublic Bus Brake System', fontsize=13, fontweight='bold')

# Draw three boxes
boxes = [
    (2, 7.2, 6, 1.4, 'Management / Regulator\n(Sets brake standards, speed limits)'),
    (2, 4.5, 6, 1.4, 'Controller\n(Driver + Safety Governor Software)'),
    (2, 1.8, 6, 1.4, 'Controlled Process\n(Vehicle Brakes / Road Surface)'),
]

for (x, y, w, h, label) in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.1',
                                    linewidth=1.5,
                                    edgecolor='black',
                                    facecolor='#d9eaf7')
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontsize=9)

# Arrow: Management -> Controller (control action)
ax.annotate('', xy=(5.5, 5.9), xytext=(5.5, 7.2),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(5.7, 6.55, 'Control\nActions', fontsize=7, color='blue')

# Arrow: Controller -> Management (feedback)
ax.annotate('', xy=(4.5, 7.2), xytext=(4.5, 5.9),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax.text(3.5, 6.55, 'Feedback', fontsize=7, color='green')

# Arrow: Controller -> Brakes (brake command)
ax.annotate('', xy=(5.5, 3.2), xytext=(5.5, 4.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(5.7, 3.85, 'Brake\nCommand', fontsize=7, color='blue')

# Arrow: Brakes -> Controller (sensor feedback)
ax.annotate('', xy=(4.5, 4.5), xytext=(4.5, 3.2),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax.text(3.5, 3.85, 'Sensor\nFeedback', fontsize=7, color='green')

plt.tight_layout()
plt.savefig('part5_stamp_diagram.png', dpi=150)
plt.close()
print("Diagram saved: part5_stamp_diagram.png")


# ─────────────────────────────────────────
# 2. STPA Hazard Table
# ─────────────────────────────────────────

print("\n--- STPA Table: Control Action = Issue Emergency Brake ---\n")

stpa_rows = [
    ("Not Provided",
     "Governor never triggers brake command",
     "Fatal accident in wet/high-speed scenario",
     "Require override if wet road + speed > 30 mph"),

    ("Provided Incorrectly",
     "Brake command sent but wrong intensity",
     "Insufficient braking, vehicle skids",
     "Validate brake force against road surface type"),

    ("Too Late",
     "Brake command delayed beyond safe window",
     "Vehicle cannot stop in time",
     "Trigger override if reaction window < 2 seconds"),

    ("Stopped Too Soon",
     "Brake released before vehicle fully stops",
     "Vehicle rolls forward, secondary collision",
     "Hold brake until speed sensor reads zero"),
]

print(f"{'Failure Mode':<22} {'What Happens':<42} {'Hazard':<40} {'Safeguard'}")
print("-" * 140)
for row in stpa_rows:
    print(f"{row[0]:<22} {row[1]:<42} {row[2]:<40} {row[3]}")


# ─────────────────────────────────────────
# 3. Safety Governor - Simple If/Else Rules
# ─────────────────────────────────────────

def safety_governor(speed_limit, road_surface, num_vehicles, prediction):
    """
    Overrides a 'Safe' prediction when road conditions look dangerous.
    Returns final decision string.
    """

    # Rule 1: Wet road and multiple vehicles near the bus
    if prediction == 'Safe' and 'Wet' in str(road_surface) and num_vehicles >= 2:
        return 'OVERRIDE - Emergency Brake'

    # Rule 2: Frost or flood is always high risk
    if 'Frost' in str(road_surface) or 'Flood' in str(road_surface):
        return 'OVERRIDE - Emergency Brake'

    # Rule 3: Too many vehicles around the bus even on dry road
    if prediction == 'Safe' and num_vehicles >= 3:
        return 'OVERRIDE - Emergency Brake'

    # No override needed
    return prediction


# ─────────────────────────────────────────
# 4. Run Governor on the False Negatives
# ─────────────────────────────────────────

print("\n--- Testing Governor on False Negatives from Part 4 ---\n")

fn = pd.read_csv('part4_false_negatives.csv', encoding='latin-1')

print(f"Total false negatives loaded: {len(fn)}")
print(f"Columns: {list(fn.columns)}\n")

# Apply governor to each row
decisions = []
for _, row in fn.iterrows():
    road_surface = row.get('Road_Surface_Conditions', 'Dry')
    num_vehicles = row.get('Number_of_Vehicles', 1)
    speed_limit  = row.get('Speed_limit', 30)
    decision = safety_governor(speed_limit, road_surface, num_vehicles, 'Safe')
    decisions.append(decision)

fn['Governor_Decision'] = decisions

overrides = sum(1 for d in decisions if d == 'OVERRIDE - Emergency Brake')
missed    = len(fn) - overrides

print(f"Governor caught (overrides): {overrides} / {len(fn)}")
print(f"Still missed by governor:    {missed} / {len(fn)}")
print(f"Override rate: {overrides / len(fn) * 100:.2f}%\n")

# Show overridden rows
override_rows = fn[fn['Governor_Decision'] == 'OVERRIDE - Emergency Brake']
if len(override_rows) > 0:
    print("Overridden rows (sample):")
    cols = ['Road_Surface_Conditions', 'Number_of_Vehicles', 'Governor_Decision']
    print(override_rows[cols].head(10).to_string(index=False))

# Save results
fn.to_csv('part5_governor_results.csv', index=False)
print("\nSaved: part5_governor_results.csv")
print("Part 5 complete.")
