import pygame
import sys

# Inicialización de Pygame
pygame.init()

# Configuraciones de la ventana
ventana_ancho, ventana_alto = 1000, 1000
ventana = pygame.display.set_mode((ventana_ancho, ventana_alto))
pygame.display.set_caption("Simulador de Navegación de Agentes")

# Colores
color_agente = (0, 255, 0)  # Verde
color_obstaculo = (255, 0, 0)  # Rojo
color_fondo = (255, 255, 255)  # Negro

def dibujar_texto(surf, texto, tamaño, x, y):
    font = pygame.font.Font(pygame.font.match_font('arial'), tamaño)
    text_surface = font.render(texto, True, (0, 0, 0))  # Texto blanco
    text_rect = text_surface.get_rect()
    text_rect.topright = (x, y)
    surf.blit(text_surface, text_rect)

def transformar_coordenadas(x, y):
    # Ajusta la escala de coordenadas a 100 píxeles por unidad
    escala = 100
    x_nuevo = ventana_ancho / 2 + x * escala
    y_nuevo = ventana_alto / 2 - y * escala
    return int(x_nuevo), int(y_nuevo)

def cargar_obstaculos(archivo):
    obstaculos = []
    with open(archivo, 'r') as f:
        for linea in f:
            partes = linea.strip().split(',')
            obstaculo_id = int(partes[0])
            vertices = []
            for i in range(1, len(partes), 2):  # Itera sobre los vértices
                # Limpia la cadena para asegurarse de que solo contenga números y el punto decimal
                x_str = partes[i].strip("() ")
                y_str = partes[i + 1].strip("() ")
                # Ahora intenta convertir a float
                x, y = float(x_str), float(y_str)
                vertices.append((x, y))
            obstaculos.append(vertices)
    return obstaculos

def dibujar_grilla(ventana, espaciado, ancho, alto, color):
    # Dibuja líneas verticales
    for x in range(0, ancho // 2, espaciado):
        pygame.draw.line(ventana, color, (ancho // 2 + x, 0), (ancho // 2 + x, alto))
        pygame.draw.line(ventana, color, (ancho // 2 - x, 0), (ancho // 2 - x, alto))
    
    # Dibuja líneas horizontales
    for y in range(0, alto // 2, espaciado):
        pygame.draw.line(ventana, color, (0, alto // 2 + y), (ancho, alto // 2 + y))
        pygame.draw.line(ventana, color, (0, alto // 2 - y), (ancho, alto // 2 - y))

    # Dibuja los ejes centrales
    pygame.draw.line(ventana, color, (ancho // 2, 0), (ancho // 2, alto))  # Eje vertical
    pygame.draw.line(ventana, color, (0, alto // 2), (ancho, alto // 2))  # Eje horizontal


def dibujar_obstaculos(ventana, obstaculos):
    for obstaculo in obstaculos:
        # Transforma y dibuja cada obstáculo como un polígono cerrado
        vertices_transformados = [transformar_coordenadas(x, y) for x, y in obstaculo]
        pygame.draw.polygon(ventana, color_obstaculo, vertices_transformados, 3)  # 3 es el grosor de la línea

def cargar_agentes(archivo):
    agentes = {}
    with open(archivo, 'r') as f:
        agente_id = 1
        for linea in f:
            x, y = map(float, linea.strip().split(','))
            x_transformado, y_transformado = transformar_coordenadas(x, y)
            agentes[agente_id] = (x_transformado, y_transformado)
            agente_id += 1
    return agentes

def cargar_movimientos(archivo):
    movimientos = {}
    with open(archivo, 'r') as f:
        for linea in f:
            # Separamos el proceso de conversión para tratar adecuadamente los valores de x y y como float
            partes = linea.strip().split(',')
            step = int(partes[0])
            agente_id = int(partes[1])
            x, y = float(partes[2]), float(partes[3])  # Tratamos x y y como float
            
            # Transformamos las coordenadas x y y aquí si es necesario, dependiendo de si deseas aplicar la transformación
            # de coordenadas en este punto o más adelante
            if step not in movimientos:
                movimientos[step] = []
            movimientos[step].append((agente_id, x, y))
    return movimientos

# Carga de obstáculos y agentes
obstaculos = cargar_obstaculos('obstaculos.txt')
# agentes = cargar_agentes('agentes.txt')
agentes = {}

movimientos_agentes = cargar_movimientos('movimientos_agentes.txt')

# Inicialización del contador de steps
step = 0

# Bucle principal de la simulación
reloj = pygame.time.Clock()

# Ajustes de la grilla
color_grilla = (200, 200, 200)  # Un gris claro para la grilla
espaciado_grilla = 100  # Cada 100 píxeles

# Bucle principal de la simulación
while True:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Fondo
    ventana.fill(color_fondo)

    # Dibuja la grilla
    dibujar_grilla(ventana, espaciado_grilla, ventana_ancho, ventana_alto, color_grilla)

    # Actualiza y dibuja agentes según el step actual
    if step in movimientos_agentes:
        for movimiento in movimientos_agentes[step]:
            print(movimiento)
            agente_id, x, y = movimiento
            # Aquí aplicamos la transformación a las nuevas posiciones antes de actualizar los agentes
            x_transformado, y_transformado = transformar_coordenadas(x, y)
            agentes[agente_id] = (x_transformado, y_transformado)  # Actualiza la posición del agente con coordenadas transformadas

    for agente_id, (x, y) in agentes.items():
        pygame.draw.circle(ventana, color_agente, (x, y), 10)  # Dibuja el agente con coordenadas ya transformadas

    # Dibuja los obstáculos
    dibujar_obstaculos(ventana, obstaculos)

    # Dibuja el contador de steps en la esquina superior derecha
    dibujar_texto(ventana, f"step: {step}", 36, ventana_ancho - 10, 10)

    # Incrementa el contador de step    
    step += 1

    # Actualiza la pantalla y espera al siguiente frame
    pygame.display.flip()
    reloj.tick(60)
