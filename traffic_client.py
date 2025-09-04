import socket
import json
import time

def send_traffic_data_to_unity(json_data):
    """Enviar datos JSON de simulación de tráfico al servidor Unity"""
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
        return True
        
    except Exception as e:
        print(f"Error conectando a Unity: {e}")
        return False

def load_and_send_from_file(filename):
    """Cargar datos JSON desde archivo y enviar a Unity"""
    try:
        with open(filename, 'r') as f:
            json_data = f.read()
        return send_traffic_data_to_unity(json_data)
    except Exception as e:
        print(f"Error cargando archivo {filename}: {e}")
        return False

if __name__ == "__main__":
    print("Cliente de Tráfico - Enviar datos a Unity")
    print("1. Asegúrate de que el servidor Unity esté ejecutándose")
    print("2. Enviando datos desde traffic_movement_data.json...")
    
    # Enviar datos desde el archivo JSON generado
    success = load_and_send_from_file('traffic_movement_data.json')
    
    if success:
        print("✅ ¡Datos enviados exitosamente a Unity!")
    else:
        print("❌ Falló el envío de datos. Asegúrate de que el servidor Unity esté ejecutándose.")
