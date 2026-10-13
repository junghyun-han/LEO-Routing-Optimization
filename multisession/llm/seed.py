# Greedy reconfiguration baseline: admit sessions in the given order; immediate route if
# one exists, otherwise switch free terminals on (fewest new links, then fewest hops).


def solve(instance, kit):
    state = kit.State(instance)
    routes = {}
    for session in instance.sessions:
        found = kit.route_session(state, session)
        if found is not None:
            state.admit(*found)
            routes[session.index] = found[0]
    return kit.finish(state, routes)
