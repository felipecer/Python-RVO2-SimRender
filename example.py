#!/usr/bin/env python

import rvo2

sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 2)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((0, 0))
a1 = sim.addAgent((1, 0))
a2 = sim.addAgent((1, 1))
a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))

# Obstacles are also supported.
o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
sim.processObstacles()

sim.setAgentPrefVelocity(a0, (1, 1))
sim.setAgentPrefVelocity(a1, (-1, 1))
sim.setAgentPrefVelocity(a2, (-1, -1))
sim.setAgentPrefVelocity(a3, (1, -1))

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

# Guarda información de obstáculos en un archivo
with open('obstaculos.txt', 'w') as archivo_obstaculos:
    for obstaculo_id in range(sim.getNumObstacleVertices()):
        # Obtiene el vértice actual y el siguiente vértice usando los índices
        vertice = sim.getObstacleVertex(obstaculo_id)
        siguiente_vertice_id = sim.getNextObstacleVertexNo(obstaculo_id)
        siguiente_vertice = sim.getObstacleVertex(siguiente_vertice_id)

        # Accede a los valores x e y directamente de las tuplas
        archivo_obstaculos.write(f'{obstaculo_id},({vertice[0]}, {vertice[1]}), ({siguiente_vertice[0]}, {siguiente_vertice[1]})\n')

print('Archivo de obstáculos generado con éxito.')

# Abre un archivo para escribir las posiciones de los agentes
with open('posiciones_agentes.txt', 'w') as archivo_posiciones:
    for step in range(1000):  # Número de steps en la simulación
        sim.doStep()
        
        for agent_no in range(sim.getNumAgents()):
            x, y = sim.getAgentPosition(agent_no)
            # Escribe la posición al archivo, omitiendo la marca de tiempo por ahora
            archivo_posiciones.write(f'{step},{agent_no + 1},{x:.2f},{y:.2f}\n')

print('Archivo de posiciones de agentes generado con éxito.')
