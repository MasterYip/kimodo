from pathlib import Path
import tempfile
import unittest
import numpy as np

from locomotion_framework.evaluation.core import evaluate_motion, expected_path, pelvis_local


def yaw_quat(yaw):
    yaw=np.asarray(yaw); out=np.zeros(yaw.shape+(4,)); out[...,0]=np.cos(yaw/2); out[...,3]=np.sin(yaw/2); return out


def world_from_local(local, origin, yaw):
    c,s=np.cos(yaw),np.sin(yaw); x=c*local[...,0]-s*local[...,1]+origin[...,0]; y=s*local[...,0]+c*local[...,1]+origin[...,1]
    return np.stack((x,y,local[...,2]+origin[...,2]),axis=-1)


class EvaluationCoreTest(unittest.TestCase):
    def test_yaw_translation_invariance(self):
        frames=12; yaw=np.linspace(-1.0,1.0,frames); origin=np.stack((np.arange(frames),-.2*np.arange(frames),np.ones(frames)),axis=-1)
        local=np.zeros((frames,2,3)); local[:,0]=[.3,.2,.1]; local[:,1]=[-.1,-.25,.4]
        world=world_from_local(local,origin[:,None,:],yaw[:,None])
        got=pelvis_local(world,origin,yaw_quat(yaw))
        np.testing.assert_allclose(got,local,atol=1e-10)

    def test_known_path_metrics_and_crossing(self):
        frames=60; fps=30.; target=expected_path(.5,0.,0.,frames,fps)
        pos=np.zeros((frames,30,3),dtype=np.float32); pos[:,:,2]=.75; pos[:,0,:2]=target
        quat=np.zeros((frames,30,4),dtype=np.float32); quat[...,0]=1
        # Shoulders establish left(+Y)/right(-Y), while wrists swap sides.
        pos[:,22,1]=.2; pos[:,23,1]=-.2; pos[:30,28,1]=.2; pos[:30,29,1]=-.2; pos[30:,28,1]=-.2; pos[30:,29,1]=.2
        pos[:,28,0]=np.sin(np.linspace(0,4*np.pi,frames))*1.0; pos[:,29,0]=-pos[:,28,0]
        joints=np.zeros((frames,29),dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"motion.npz"; np.savez(path,fps=np.array([fps],np.float32),joint_pos=joints,body_pos_w=pos,body_quat_w=quat)
            row,_=evaluate_motion(path,{"vx":"0.5","vy":"0","wz":"0"})
        self.assertLess(row["endpoint_error_m"],1e-6)
        self.assertLess(row["path_rmse_m"],1e-6)
        self.assertGreater(row["wrist_order_cross_fraction"],.45)
        self.assertEqual(row["dominant_swing_axis"],"forward")
        self.assertLess(row["swing_alignment_to_command_deg"], 20.0)


if __name__ == "__main__": unittest.main()
