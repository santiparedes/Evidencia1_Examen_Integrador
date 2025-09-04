# Simulación de Tráfico - Integración con Unity

Sistema que convierte una simulación de intersección de tráfico en una visualización en tiempo real en Unity usando comunicación TCP/IP.

## Descripción del Sistema

1. **Simulación Python** (`traffic_sim_json.py`) - Ejecuta la simulación de tráfico y genera datos de movimiento en JSON
2. **Cliente de Comunicación** (`traffic_client.py`) - Envía datos JSON a Unity vía TCP/IP
3. **Servidor Unity** (`TCPIPServerAsync.cs`) - Recibe datos JSON y los traduce en movimientos de objetos 3D

## Requisitos

### Python
```bash
pip install agentpy numpy
```

### Unity
- Instalar el paquete **Newtonsoft.Json** desde Unity Package Manager
- Descargar desde: https://github.com/JamesNK/Newtonsoft.Json

## Configuración de Unity

### 1. Configuración de Escena
1. Crear un GameObject vacío llamado "TrafficController"
2. Adjuntar el script `TCPIPServerAsync` a este GameObject
3. Crear prefabs de autos (cubos simples funcionan bien)
4. Crear prefabs de semáforos (esferas de colores funcionan bien)
5. Crear un GameObject vacío en el centro de la intersección

### 2. Configuración del Inspector
En el componente TCPIPServerAsync, asignar:
- **Car Prefab**: Tu modelo de auto prefab
- **Traffic Light Prefab**: Tu modelo de semáforo prefab
- **Intersection Center**: Transform en el centro de tu intersección

### 3. Configuración de Red
- **IP**: 127.0.0.1 (localhost)
- **Puerto**: 1101
- Asegúrate de que el firewall no bloquee este puerto

## Cómo Ejecutar

### Opción 1: Sistema Completo (Recomendado)
```bash
# 1. Primero iniciar Unity y reproducir la escena
# 2. Luego ejecutar:
python traffic_sim_json.py
```

### Opción 2: Solo Prueba de Conexión
```bash
# Terminal 1 - Servidor de prueba
python server-duplex.py

# Terminal 2 - Cliente de prueba
python client-duplex.py
```

### Opción 3: Prueba del Sistema
```bash
python test_system.py
```

## Parámetros de Simulación

Editar `params` en `traffic_sim_json.py`:
```python
params = {
    'steps': 600,          # Duración de simulación
    'lambda_S': 0.10,      # Tasa de llegada de autos desde Sur
    'lambda_E': 0.10,      # Tasa de llegada de autos desde Este
    'lambda_W': 0.10,      # Tasa de llegada de autos desde Oeste
    'v_free': 7.0,         # Velocidad libre (m/s)
    'policy': 'adaptive',  # Política de semáforos
}
```

## Formato de Datos

Los datos JSON contienen pasos de tiempo con:
```json
{
  "timestep": 0,
  "traffic_lights": {"S": "R", "E": "G", "W": "G"},
  "cars": [
    {
      "id": "S_0_0",
      "origin": "S",
      "position": {"x": 1.75, "y": -80.0},
      "direction": {"x": 0, "y": 1},
      "state": "approach",
      "turn": "L"
    }
  ]
}
```

## Solución de Problemas

### Error "Connection refused"
- Asegúrate de que Unity esté ejecutándose y la escena esté reproduciéndose
- Verifica que el script TCPIPServerAsync esté adjunto y activo

### No aparecen autos en Unity
- Verifica que carPrefab esté asignado en el inspector
- Revisa que intersectionCenter esté configurado correctamente
- Mira la Consola de Unity para mensajes de error

### Los semáforos no cambian
- Verifica que trafficLightPrefab esté asignado
- Asegúrate de que los objetos de semáforo tengan componentes Renderer

### Errores de parsing JSON
- Asegúrate de que Newtonsoft.Json esté instalado en Unity
- Revisa la Consola de Unity para mensajes de error detallados

## Notas de Rendimiento

- La simulación genera ~600 pasos de tiempo por defecto
- Cada paso contiene datos de posición para todos los autos activos
- El archivo JSON típicamente es de 1-5 MB dependiendo de la densidad de tráfico
- La reproducción en Unity puede acelerarse reduciendo `playbackSpeed`

## Archivos del Proyecto

- `traffic_sim_json.py` - Simulación principal de tráfico
- `traffic_client.py` - Cliente para enviar datos a Unity
- `test_system.py` - Script de prueba del sistema
- `server-duplex.py` - Servidor de prueba simple
- `client-duplex.py` - Cliente de prueba simple
- `TCPIPServerAsync.cs` - Script de Unity para recibir datos
