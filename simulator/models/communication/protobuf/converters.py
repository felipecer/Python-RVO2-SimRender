from typing import Tuple
from simulator.models.communication import messages as orig
from simulator.models.communication.protobuf import sim_protobuf_messages as pb

def to_protobuf_and_type(msg: orig.BaseMessage) -> Tuple[str, bytes]:
    """
    Given an original dataclass-based message, return a Tuple:
    - string name of the Protobuf message type
    - the serialized protobuf binary
    """
    if isinstance(msg, orig.SimulationInitializedMessage):
        proto = to_simulation_initialized_message(msg)
        return "SimulationInitializedMessage", proto.serialize()
    elif isinstance(msg, orig.AllGoalsProcessedMessage):
        proto = to_all_goals_processed_message(msg)
        return "AllGoalsProcessedMessage", proto.serialize()
    elif isinstance(msg, orig.GoalPositionUpdatedMessage):
        proto = to_goal_position_updated_message(msg)
        return "GoalPositionUpdatedMessage", proto.serialize()
    elif isinstance(msg, orig.AgentsStateUpdateMessage):
        proto = to_agents_state_update_message(msg)
        return "AgentsStateUpdateMessage", proto.serialize()
    elif isinstance(msg, orig.ObstaclesProcessedMessage):
        proto = to_obstacles_processed_message(msg)
        return "ObstaclesProcessedMessage", proto.serialize()
    elif isinstance(msg, orig.NewObstacleAddedMessage):
        proto = to_new_obstacle_added_message(msg)
        return "NewObstacleAddedMessage", proto.serialize()
    elif isinstance(msg, orig.RayCastingUpdateMessage):
        proto = to_ray_casting_update_message(msg)
        return "RayCastingUpdateMessage", proto.serialize()
    else:
        raise ValueError(f"Unsupported message type: {type(msg)}")


def to_vector2(t: Tuple[float, float]) -> pb.Vector2:
    return pb.Vector2(x=t[0], y=t[1])

def from_vector2(v: pb.Vector2) -> Tuple[float, float]:
    return (v.x, v.y)


def to_agent_goal(goal: orig.AgentGoal) -> pb.AgentGoal:
    return pb.AgentGoal(agent_id=goal.agent_id, goal=to_vector2(goal.goal))

def from_agent_goal(goal: pb.AgentGoal) -> orig.AgentGoal:
    return orig.AgentGoal(agent_id=goal.agent_id, goal=from_vector2(goal.goal))


def to_agent_state(state: orig.AgentState) -> pb.AgentState:
    return pb.AgentState(
        agent_id=state.agent_id,
        position=to_vector2(state.position),
        velocity=to_vector2(state.velocity),
        preferred_velocity=to_vector2(state.preferred_velocity),
        distance_to_goal=state.distance_to_goal
    )

def from_agent_state(state: pb.AgentState) -> orig.AgentState:
    return orig.AgentState(
        agent_id=state.agent_id,
        position=from_vector2(state.position),
        velocity=from_vector2(state.velocity),
        preferred_velocity=from_vector2(state.preferred_velocity),
        distance_to_goal=state.distance_to_goal
    )


def to_agent_init(data: orig.AgentInitData) -> pb.AgentInitData:
    return pb.AgentInitData(
        agent_id=data.agent_id,
        radius=data.radius,
        max_speed=data.max_speed,
        neighbor_dist=data.neighbor_dist,
        max_neighbors=data.max_neighbors,
        time_horizon=data.time_horizon,
        time_horizon_obst=data.time_horizon_obst,
        goal=to_vector2(data.goal),
        behaviour=data.behaviour or ""
    )

def from_agent_init(data: pb.AgentInitData) -> orig.AgentInitData:
    return orig.AgentInitData(
        agent_id=data.agent_id,
        radius=data.radius,
        max_speed=data.max_speed,
        neighbor_dist=data.neighbor_dist,
        max_neighbors=data.max_neighbors,
        time_horizon=data.time_horizon,
        time_horizon_obst=data.time_horizon_obst,
        goal=from_vector2(data.goal),
        behaviour=data.behaviour or None
    )

def to_obstacle(obs: orig.Obstacle) -> pb.Obstacle:
    return pb.Obstacle(vertices=[to_vector2(v) for v in obs.vertices])

def from_obstacle(obs: pb.Obstacle) -> orig.Obstacle:
    return orig.Obstacle(vertices=[from_vector2(v) for v in obs.vertices])

def to_ray_hit(hit: orig.RayHit) -> pb.RayHit:
    return pb.RayHit(x=hit.x or 0.0, y=hit.y or 0.0)

def from_ray_hit(hit: pb.RayHit) -> orig.RayHit:
    return orig.RayHit(x=hit.x, y=hit.y)

def to_simulation_initialized_message(msg: orig.SimulationInitializedMessage) -> pb.SimulationInitializedMessage:
    return pb.SimulationInitializedMessage(
        step=msg.step,
        agent_initialization_data=[to_agent_init(a) for a in msg.agent_initialization_data],
        obstacles=[to_obstacle(o) for o in msg.obstacles],
        goals=[to_agent_goal(g) for g in msg.goals]
    )

def from_simulation_initialized_message(msg: pb.SimulationInitializedMessage) -> orig.SimulationInitializedMessage:
    return orig.SimulationInitializedMessage(
        step=msg.step,
        agent_initialization_data=[from_agent_init(a) for a in msg.agent_initialization_data],
        obstacles=[from_obstacle(o) for o in msg.obstacles],
        goals=[from_agent_goal(g) for g in msg.goals]
    )


def to_agents_state_update_message(msg: orig.AgentsStateUpdateMessage) -> pb.AgentsStateUpdateMessage:
    return pb.AgentsStateUpdateMessage(
        step=msg.step,
        agent_state_list=[to_agent_state(a) for a in msg.agent_state_list]
    )

def from_agents_state_update_message(msg: pb.AgentsStateUpdateMessage) -> orig.AgentsStateUpdateMessage:
    return orig.AgentsStateUpdateMessage(
        step=msg.step,
        agent_state_list=[from_agent_state(a) for a in msg.agent_state_list]
    )


def to_goal_position_updated_message(msg: orig.GoalPositionUpdatedMessage) -> pb.GoalPositionUpdatedMessage:
    return pb.GoalPositionUpdatedMessage(
        step=msg.step,
        goals=[to_agent_goal(g) for g in msg.goals]
    )

def from_goal_position_updated_message(msg: pb.GoalPositionUpdatedMessage) -> orig.GoalPositionUpdatedMessage:
    return orig.GoalPositionUpdatedMessage(
        step=msg.step,
        goals=[from_agent_goal(g) for g in msg.goals]
    )


def to_obstacles_processed_message(msg: orig.ObstaclesProcessedMessage) -> pb.ObstaclesProcessedMessage:
    return pb.ObstaclesProcessedMessage(
        step=msg.step,
        obstacles=[to_obstacle(o) for o in msg.obstacles]
    )

def from_obstacles_processed_message(msg: pb.ObstaclesProcessedMessage) -> orig.ObstaclesProcessedMessage:
    return orig.ObstaclesProcessedMessage(
        step=msg.step,
        obstacles=[from_obstacle(o).vertices for o in msg.obstacles]  # original expects List[List[Tuple]]
    )


def to_all_goals_processed_message(msg: orig.AllGoalsProcessedMessage) -> pb.AllGoalsProcessedMessage:
    return pb.AllGoalsProcessedMessage(
        step=msg.step,
        goals=[to_agent_goal(g) for g in msg.goals]
    )

def from_all_goals_processed_message(msg: pb.AllGoalsProcessedMessage) -> orig.AllGoalsProcessedMessage:
    return orig.AllGoalsProcessedMessage(
        step=msg.step,
        goals=[from_agent_goal(g) for g in msg.goals]
    )


def to_new_obstacle_added_message(msg: orig.NewObstacleAddedMessage) -> pb.NewObstacleAddedMessage:
    return pb.NewObstacleAddedMessage(
        step=msg.step,
        obstacle=to_obstacle(orig.Obstacle(vertices=msg.obstacle))
    )

def from_new_obstacle_added_message(msg: pb.NewObstacleAddedMessage) -> orig.NewObstacleAddedMessage:
    return orig.NewObstacleAddedMessage(
        step=msg.step,
        obstacle=[from_vector2(v) for v in msg.obstacle.vertices]
    )


def to_ray_casting_update_message(msg: orig.RayCastingUpdateMessage) -> pb.RayCastingUpdateMessage:
    return pb.RayCastingUpdateMessage(
        step=msg.step,
        agent_id=msg.agent_id,
        hits=[to_ray_hit(h) for h in msg.hits]
    )

def from_ray_casting_update_message(msg: pb.RayCastingUpdateMessage) -> orig.RayCastingUpdateMessage:
    return orig.RayCastingUpdateMessage(
        step=msg.step,
        agent_id=msg.agent_id,
        hits=[from_ray_hit(h) for h in msg.hits]
    )