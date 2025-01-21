from DataInterface import DataInterface

data_interface = DataInterface(
    left_arm_start_joint__positions=tuple([-0.09918270446100141, -1.5253831709203354, 2.1674539606378547,
                                           4.03517106103823, -1.5213525784559918, 2.023743354909077])
)
# data_interface.startDataCollection(remove_zero_actions=True, collection_freq_hz=10)
data_interface.replayTrajectory(traj_file_path='/home/weirdlab/ur_bc/data/traj_18.npz', joint_position_replay=True)


"""
[0.09600820791777794, -1.9700557395947929, 2.2113284892461644, 4.1397435859649026,
-1.5766150033324582, 2.3817672802827907]
False

[ 0.1001, -1.9640,  2.1928,  4.1546, -1.5883,  2.3854,  0.0000]
"""
