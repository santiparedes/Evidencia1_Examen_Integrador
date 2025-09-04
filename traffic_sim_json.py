import agentpy as ap
import numpy as np
import json
import socket
import time

# Parameters (same as original)
params = {
    'steps': 600,          # duración en ticks (1 tick = 1 s)
    'green_ns': 20,        # VERDE para Norte-Sur
    'green_ew': 20,        # VERDE para Este-Oeste
    'yellow': 3,           # ÁMBAR
    'all_red': 1,          # ALL-RED (despeje)
    # Tasas Poisson de arribo (veh/s) por aproximación
    'lambda_S': 0.10,
    'lambda_E': 0.10,
    'lambda_W': 0.10,
    # Cinemática
    'v_free': 7.0,         # m/s
    'headway': 13.0,       # m separación mínima (increased for more spacing)
    # Geometría (intersección centrada en 0,0)
    'L': 80.0,             # media-calzada (desde centro al extremo de dibujo)
    'w': 7.0,              # ancho de carril (increased for more lateral spacing)
    # Probabilidades de giro/recto (deben sumar 1 para cada origen)
    'p_S_left': 0.5,
    'p_S_right': 0.5,
    'p_S_straight': 0.0,
    'p_E_right': 0.0,  # East cars only go straight (East)
    'p_E_straight': 1.0,
    'p_W_right': 0.5,  # West cars turn right (South)
    'p_W_straight': 0.5, # West cars go straight (East)

    # Política: 'fixed' (actual) o 'adaptive' (heurística)
    'policy': 'adaptive',

    # Ventanas mín./máx. de verde por grupo de fases (ajusta a tu gusto)
    'gmin_ns': 8,  'gmax_ns': 50,   # Nota: en tu T-cruce "NS" = fase de S
    'gmin_ew': 8,  'gmax_ew': 50,   # "EW" = fase de E/W

    # Umbral de cola para extender (si cola verde >= cola roja + theta → extiende)
    'theta': 2
}

class FourWaySignals(ap.Agent):
    """Control alternado: fase NS y fase EW, con subestados G/Y/AR."""

    def setup(self, green_ns, green_ew, yellow, all_red):
        self.g_ns, self.g_ew = int(green_ns), int(green_ew)
        self.y, self.ar = int(yellow), int(all_red)
        self.phase = 0          # 0 = NS verde, 1 = EW verde
        self.sub = 'G'          # 'G','Y','AR'
        self.t_in = 0
        self.timeline = []      # [(t, {'N':R/Y/G, 'S':..., 'E':..., 'W':...})]

    def lights(self):
      L = {d:'R' for d in ['S','E','W']}
      if self.phase == 0:
        L['E'] = L['W'] = self.sub
      else:
        L['S'] = self.sub
      return L

    @property
    def green_dirs(self):
        if self.sub != 'G': return set()
        return {'S'} if self.phase == 0 else {'E','W'}

    def step(self):
        # Log opcional
        self.timeline.append((self.model.t, self.lights()))

        # Atajos de tiempos
        if self.phase == 0:    # E/W
            gmin, gmax = self.model.p.gmin_ew, self.model.p.gmax_ew
        else:                  # S
            gmin, gmax = self.model.p.gmin_ns, self.model.p.gmax_ns

        # Política adaptativa durante 'G'
        if self.sub == 'G' and getattr(self.model.p, 'policy', 'fixed') == 'adaptive':
            # colas por grupo verde vs. rojo
            qs = self.model.queues_by_dir()
            if self.phase == 0:
                q_green = qs['E'] + qs['W']
                q_red   = qs['S']
            else:
                q_green = qs['S']
                q_red   = qs['E'] + qs['W']

            # Reglas: respetar gmin/gmax; si gmin cumplido, decide extender o cambiar
            if self.t_in < gmin:
                self.t_in += 1
            elif self.t_in >= gmax:
                self.sub, self.t_in = 'Y', 0
            elif q_green >= q_red + self.model.p.theta:
                # Extiende verde
                self.t_in += 1
            else:
                # Cambia a amarillo
                self.sub, self.t_in = 'Y', 0
            return  # importante: no caigas al plan fijo

        # --- Plan fijo (tu lógica original) ---
        if self.phase == 0:
            if self.sub == 'G' and self.t_in >= self.g_ew: self.sub, self.t_in = 'Y', 0
            elif self.sub == 'Y' and self.t_in >= self.y: self.sub, self.t_in = 'AR', 0
            elif self.sub == 'AR' and self.t_in >= self.ar: self.phase, self.sub, self.t_in = 1, 'G', 0
            else: self.t_in += 1
        else:
            if self.sub == 'G' and self.t_in >= self.g_ns: self.sub, self.t_in = 'Y', 0
            elif self.sub == 'Y' and self.t_in >= self.y: self.sub, self.t_in = 'AR', 0
            elif self.sub == 'AR' and self.t_in >= self.ar: self.phase, self.sub, self.t_in = 0, 'G', 0
            else: self.t_in += 1

class Car(ap.Agent):
    """Auto en carril recto/giros: entra desde S/E/W y desde S gira a la izquierda o derecha."""

    def setup(self, origin):
        self.wait = 0           # segundos acumulados detenido
        self.origin = origin            # 'S','E','W'   (sin 'N')
        self.state = 'approach'         # 'stop','go','done'
        self.v = self.model.p.v_free
        L, w = self.model.p.L, self.p.w
        off = w/2

        self.turn = None        # 'L', 'R', or 'S' (for straight)
        self.turned = False     # bandera para no girar más de una vez
        self.car_id = f"{origin}_{self.model.t}_{len(self.model.cars)}"  # Unique ID

        if origin == 'S':
            # Spawnea en el carril que sube hacia la intersección
            self.pos = np.array([ +off, -L ])
            self.dir = np.array([ 0, +1 ])
            self.stopline = np.array([ +off, -12 ])  # Más atrás 
            # Decidir giro (izquierda = hacia W, derecha = hacia E, recto = hacia N)
            p_left = getattr(self.model.p, 'p_S_left', 1/3)
            p_right = getattr(self.model.p, 'p_S_right', 1/3)
            p_straight = getattr(self.model.p, 'p_S_straight', 1/3)
            choice = np.random.choice(['L', 'R', 'S'], p=[p_left, p_right, p_straight])
            self.turn = choice
            # The goal depends on the turn:
            if self.turn == 'R':
                # Salida hacia +x por el carril inferior (y = -off)
                self.goal = np.array([ +L, -off ])
            elif self.turn == 'L':
                # Salida hacia -x por el carril superior (y = +off)
                self.goal = np.array([ -L, +off ])
            else: # Straight
                # Salida hacia +y por el carril superior (x = +off)
                self.goal = np.array([ +off, +L ])

        elif origin == 'E':
            # Entra desde el Este y solo va recto (hacia W)
            self.pos = np.array([ +L, +off ])
            self.dir = np.array([ -1, 0 ])
            self.stopline = np.array([ +12, +off ])  # Más atrás  
            self.turn = 'S' # East cars only go straight
            # Salida hacia -x por el carril superior (y = +off)
            self.goal = np.array([ -L, +off ])

        else:  # 'W'
            # Entra desde el Oeste y va recto (hacia E) o gira a la derecha (hacia S)
            self.pos = np.array([ -L, -off ])
            self.dir = np.array([ +1, 0 ])
            self.stopline = np.array([ -12, -off ])  # Más atrás 
            # Decidir giro (recto = hacia E, derecha = hacia S)
            p_straight = getattr(self.model.p, 'p_W_straight', 0.5)
            p_right = getattr(self.model.p, 'p_W_right', 0.5)
            choice = np.random.choice(['S', 'R'], p=[p_straight, p_right])
            self.turn = choice
            if self.turn == 'R':
                # Salida hacia -y por el carril izquierdo (x = -off)
                self.goal = np.array([ -off, -L ])
            else: # Straight
                 # Salida hacia +x por el carril inferior (y = -off)
                self.goal = np.array([ +L, -off ])

    def dist_to(self, p):
        return np.linalg.norm(self.pos - p)

    def step(self):
        if self.state == 'done':
            return

        # Si llegó a la meta, termina
        if self.dist_to(self.goal) < 8.0:
            self.state = 'done'
            return

        # Zona de decisión cerca de la stopline (20 m para detectar mejor con stop lines más lejanas)
        near = self.dist_to(self.stopline) < 20.0

        # Reglas de luz (diccionario por origen: 'S','E','W')
        lights = self.model.ctrl.lights() # Get the current lights state
        if near and lights[self.origin] != 'G': # Stop if near stopline and light is not Green
            self.state = 'stop'
            self.wait += 1
            return
        else:
            self.state = 'go'

        # Espacio de seguridad con el líder en el mismo carril
        vmax = self.v
        head = self.model.headway_ahead(self)
        if head is not None:
            gap = np.linalg.norm(head.pos - self.pos)
            if gap < self.model.p.headway:
                vmax = 0.0

        # --- Lógica de giro para vehículos ---
        # Giramos al cruzar el centro de la intersección (y ≳ 0 para S, x ≲ 0 para E, x ≳ 0 para W)
        if not self.turned:
            # Posición tentativa para este paso
            next_pos = self.pos + self.dir * vmax * 1.0  # dt = 1 s
            # Comprobamos cruce del centro
            crossed_center = False
            if self.origin == 'S' and next_pos[1] >= 0.0:
                crossed_center = True
            elif self.origin == 'E' and next_pos[0] <= 0.0:
                crossed_center = True
            elif self.origin == 'W' and next_pos[0] >= 0.0:
                crossed_center = True

            if crossed_center:
                # Realiza el giro y alinea al carril de salida
                if self.turn == 'R':
                    if self.origin == 'S':
                        # Salida hacia el Este: dirección +x, carril y = -off
                        self.dir = np.array([ +1, 0 ])
                        self.pos = np.array([ self.pos[0], -self.model.p.w/2 ])
                    elif self.origin == 'E':
                        # Salida hacia el Sur: dirección -y, carril x = +off
                        self.dir = np.array([ 0, -1 ])
                        self.pos = np.array([ +self.model.p.w/2, self.pos[1] ])
                    else: # 'W'
                        # Salida hacia el Sur: dirección -y, carril x = -off
                        self.dir = np.array([ 0, -1 ])
                        self.pos = np.array([ -self.model.p.w/2, self.pos[1] ])
                elif self.turn == 'L':
                    # Salida hacia el Oeste: dirección -x, carril y = +off
                    self.dir = np.array([ -1, 0 ])
                    self.pos = np.array([ self.pos[0], +self.model.p.w/2 ])
                # No need to change direction for 'S' (straight)
                self.turned = True

        # Avanzar
        self.pos = self.pos + self.dir * vmax * 1.0  # dt=1 s

class FourWayModel(ap.Model):

    def setup(self):
        p = self.p
        self.ctrl = FourWaySignals(self, p.green_ns, p.green_ew, p.yellow, p.all_red)
        self.cars = ap.AgentList(self, 0, Car)
        self.spawn_counts = {d:0 for d in ['S','E','W']}
        # Log para análisis si quieres después
        self.log = []
        self.t = 0  # reloj simple para corridas sin animación
        self.metrics = {
            'throughput': 0,
            'delay_sum': 0,
            'delay_count': 0,
            'qmax': {'S': 0, 'E': 0, 'W': 0}
        }
        # New: Store movement data for JSON export
        self.movement_data = []

    def headway_ahead(self, me):
        """Líder en el mismo carril y sentido, si existe."""
        same = [c for c in self.cars if c is not me and np.allclose(c.dir, me.dir)]
        if not same: return None
        # candidato delante si el vector a él está en dirección de me.dir y más adelante
        ahead = []
        for c in same:
            v = c.pos - me.pos
            proj = np.dot(v, me.dir)
            if proj > 0:  # adelante
                ahead.append((proj, c))
        if not ahead: return None
        return min(ahead, key=lambda x: x[0])[1]

    def spawn_poisson(self, origin, lam):
        k = np.random.poisson(lam)
        for _ in range(k):
            self.cars.append(Car(self, origin=origin))
            self.spawn_counts[origin]+=1

    def queues_by_dir(self):
        qs = {'S': 0, 'E': 0, 'W': 0}
        for c in self.cars:
            if c.state == 'stop':
                qs[c.origin] += 1
        return qs

    def step(self):
        # 1) arribos
        self.spawn_poisson('S', self.p.lambda_S)
        self.spawn_poisson('E', self.p.lambda_E)
        self.spawn_poisson('W', self.p.lambda_W)

        # 2) señales
        self.ctrl.step()

        # 3) autos
        self.cars.step()

        # --- métricas por paso ---
        qs = self.queues_by_dir()
        for d in qs:
            self.metrics['qmax'][d] = max(self.metrics['qmax'][d], qs[d])

        # contabilidad de 'done' y delays asociados
        done = [c for c in self.cars if c.state == 'done']
        self.metrics['throughput'] += len(done)
        for c in done:
            self.metrics['delay_sum'] += c.wait
            self.metrics['delay_count'] += 1

        # NEW: Capture movement data for this timestep
        timestep_data = {
            'timestep': self.t,
            'traffic_lights': self.ctrl.lights(),
            'cars': []
        }
        
        for car in self.cars:
            if car.state != 'done':
                car_data = {
                    'id': car.car_id,
                    'origin': car.origin,
                    'position': {'x': float(car.pos[0]), 'y': float(car.pos[1])},
                    'direction': {'x': float(car.dir[0]), 'y': float(car.dir[1])},
                    'state': car.state,
                    'turn': car.turn,
                    'turned': car.turned,
                    'wait_time': car.wait
                }
                timestep_data['cars'].append(car_data)
        
        self.movement_data.append(timestep_data)

        # 4) limpieza de autos terminados (opcional)
        self.cars = ap.AgentList(self, [c for c in self.cars if c.state != 'done'], Car)
        self.t += 1

    def get_movement_json(self):
        """Return the movement data as JSON string"""
        return json.dumps(self.movement_data, indent=2)

def run_simulation_and_send_to_unity():
    """Ejecutar la simulación de tráfico y enviar resultados a Unity"""
    print("Iniciando simulación de tráfico...")
    
    # Ejecutar simulación
    model = FourWayModel(params)
    model.run()
    
    print(f"Simulación completada. Generados {len(model.movement_data)} pasos de tiempo")
    print(f"Total de autos procesados: {model.metrics['throughput']}")
    
    # Obtener datos JSON
    json_data = model.get_movement_json()
    
    # Enviar a Unity
    try:
        print("Conectando al servidor Unity...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", 1101))
        
        # Recibir mensaje inicial de Unity
        from_server = s.recv(4096)
        print("Recibido del servidor Unity:", from_server.decode("ascii"))
        
        # Enviar confirmación
        s.send(b"Traffic simulation data ready")
        
        # Enviar datos JSON
        print("Enviando datos de movimiento a Unity...")
        s.send(json_data.encode('utf-8'))
        
        # Enviar marcador de fin
        s.send(b"$")
        
        print("¡Datos enviados exitosamente!")
        s.close()
        
    except Exception as e:
        print(f"Error conectando a Unity: {e}")
        print("Guardando datos en archivo...")
        with open('traffic_movement_data.json', 'w') as f:
            f.write(json_data)
        print("Datos guardados en traffic_movement_data.json")

if __name__ == "__main__":
    run_simulation_and_send_to_unity()
