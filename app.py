from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def sim_page():
    return render_template('sim.html')

@app.route("/home")
def home_page():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        charges = []
        num_charges = int(request.form['num_charges'])
        for i in range(num_charges):
            x = float(request.form[f'x{i}'])
            y = float(request.form[f'y{i}'])
            q = float(request.form[f'q{i}'])
            charges.append((x, y, q))
        img, field_values = generate_plot(charges)
        return render_template('sim.html', img_data=img, field_values=field_values)
    return render_template('sim.html', img_data=None, field_values=None)


def generate_plot(Q):
    x1, y1 = -6, -6
    x2, y2 = 6, 6
    lres = 10
    m, n = lres * (y2 - y1), lres * (x2 - x1)
    x, y = np.linspace(x1, x2, n), np.linspace(y1, y2, m)
    x, y = np.meshgrid(x, y)
    Ex = np.zeros((m, n))
    Ey = np.zeros((m, n))

    k = 9 * 10**9  # Coulomb's constant

    # Calculate electric field at each grid point
    for j in range(m):
        for i in range(n):
            xp, yp = x[j][i], y[j][i]
            for q in Q:
                deltaX = xp - q[0]
                deltaY = yp - q[1]
                distance = np.sqrt(deltaX**2 + deltaY**2)

                if distance != 0:
                    E = (k * q[2]) / (distance**2)
                    Ex[j][i] += E * (deltaX / distance)
                    Ey[j][i] += E * (deltaY / distance)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Plot charges with different colors based on the sign of the charge
    for q in Q:
        color = 'green' if q[2] > 0 else 'red'
        ax.scatter(q[0], q[1], c=color, s=abs(q[2]) * 50, zorder=1)
        ax.text(q[0] + 0.1, q[1] - 0.3, f'{q[2]} unit', color='black', zorder=2)

    ax.streamplot(x, y, Ex, Ey, linewidth=1, density=1.5, zorder=0)
    plt.title('Electrostatic Field Simulation')

    # Calculate net electric field at each charge location considering all other charges
    field_values = []
    for q in Q:
        q_x, q_y, q_mag = q
        total_Ex, total_Ey = 0, 0
        for other_q in Q:
            if other_q != q:
                deltaX = q_x - other_q[0]
                deltaY = q_y - other_q[1]
                distance = np.sqrt(deltaX**2 + deltaY**2)
                if distance != 0:
                    E = (k * abs(other_q[2])) / (distance**2)
                    total_Ex += E * (deltaX / distance)
                    total_Ey += E * (deltaY / distance)

        # Net electric field magnitude and direction
        net_E = np.sqrt(total_Ex**2 + total_Ey**2)
        field_values.append(f"Charge at ({q_x}, {q_y}): |E| = {net_E:.2e} N/C")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)  # Close the figure to prevent memory leaks

    return img_data, field_values


if __name__ == '__main__':
    app.run(debug=True)