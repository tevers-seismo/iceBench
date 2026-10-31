import iceBench

# Build Associator

# PARAMETERS for pyocto.VelocityModel0D
vpvs_ratio = 1.9
vp = 3.8
vs = vp / vpvs_ratio

velocity = {
    "p_velocity": vp,  # km/s, ice
    "s_velocity": vs,  # km/s
    "tolerance": 0.1,  # travel time tolerance in [s]
}  # we skip association_cutoff_distance here for now

# PARAMETERS for pyocto.OctoAssociator.from_area
params = {
    "lat": (45.92, 45.95),  # source region incl. the out-of-network SE events
    "lon": (7.83, 7.86),
    "zlim": (-3.95, -2.9),  # km, z DOWN -> stations at ~-3.7;
    "time_before": 2.0,  # s, overlap between consecutive time slices
    "min_node_size": 0.05,  # km, resolve the 100-200 m station spacing; if a node becomes smaller the event creation and localisation is triggered
    "min_node_size_location": 0.01,  # km; precision regarding discretisation of the location grid
    "pick_match_tolerance": 0.2,  # s; Maximum difference between the predicted travel time and the observed time for associating a pick to an origin
    "min_interevent_time": 0.5,  # s, minimum time required between two events
    "min_pick_fraction": 0.0,  # docs: disable for small deployments
    "n_picks": 6,  # Minumum required number of picks for an event.
    "n_p_picks": 2,  # Minumum required number of P picks for an event
    "n_s_picks": 2,  # Minumum required number of s picks for an event
    "n_p_and_s_picks": 2,  # Minumum required number of stations that have both P and S pick for an event
    # "refinement_iterations" :x,      # The number of localisation and pick matching iterations.
    # time_slicing:                     # The size of each time block.
}

associator = iceBench.associate.build_associator(params, velocity)
