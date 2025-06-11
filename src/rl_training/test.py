import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class TransformationTester:
    def __init__(self):
        pass
    
    def _mujoco_quat_to_scipy(self, mj_quat):
        """Convert MuJoCo quaternion [w,x,y,z] to scipy format [x,y,z,w]"""
        return np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]])

    def _scipy_quat_to_mujoco(self, scipy_quat):
        """Convert scipy quaternion [x,y,z,w] to MuJoCo format [w,x,y,z]"""
        return np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])

    def _get_yaw_from_quat(self, mj_quat):
        """Extract yaw from MuJoCo quaternion"""
        try:
            scipy_quat = self._mujoco_quat_to_scipy(mj_quat)
            scipy_quat = scipy_quat / np.linalg.norm(scipy_quat)
            euler = R.from_quat(scipy_quat).as_euler('ZYX', degrees=False)
            return euler[0]  # Return yaw (Z rotation)
        except (ValueError, np.linalg.LinAlgError):
            return 0.0

    def _wrap_angle(self, angle):
        """Normalize angle to range [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def test_identity_quaternion(self):
        """Test identity quaternion - should give zero rotations"""
        print("=== TEST 1: Identity Quaternion ===")
        mj_quat = np.array([1.0, 0.0, 0.0, 0.0])  # MuJoCo identity
        scipy_quat = self._mujoco_quat_to_scipy(mj_quat)
        
        print(f"MuJoCo quat (w,x,y,z): {mj_quat}")
        print(f"Scipy quat (x,y,z,w): {scipy_quat}")
        
        # Test conversion back
        mj_back = self._scipy_quat_to_mujoco(scipy_quat)
        print(f"Back to MuJoCo: {mj_back}")
        print(f"Conversion error: {np.linalg.norm(mj_quat - mj_back)}")
        
        # Test Euler angles
        euler_zyx = R.from_quat(scipy_quat).as_euler('ZYX', degrees=True)
        euler_xyz = R.from_quat(scipy_quat).as_euler('XYZ', degrees=True)
        
        print(f"Euler ZYX (yaw,pitch,roll) [deg]: {euler_zyx}")
        print(f"Euler XYZ (roll,pitch,yaw) [deg]: {euler_xyz}")
        
        # Test yaw extraction
        yaw = self._get_yaw_from_quat(mj_quat)
        print(f"Extracted yaw [deg]: {np.degrees(yaw)}")
        
        print()
    
    def test_pure_rotations(self):
        """Test pure rotations around each axis"""
        print("=== TEST 2: Pure Rotations ===")
        
        test_angles = [30, 45, 90, -30, -45, -90]  # degrees
        
        for angle_deg in test_angles:
            angle_rad = np.radians(angle_deg)
            print(f"\n--- Testing {angle_deg}° rotations ---")
            
            # Pure yaw rotation (Z-axis)
            r_yaw = R.from_euler('Z', angle_rad)
            scipy_quat_yaw = r_yaw.as_quat()
            mj_quat_yaw = self._scipy_quat_to_mujoco(scipy_quat_yaw)
            
            extracted_yaw = self._get_yaw_from_quat(mj_quat_yaw)
            yaw_error = abs(angle_rad - extracted_yaw)
            
            print(f"Yaw {angle_deg}°: MJ_quat={mj_quat_yaw}")
            print(f"  Extracted yaw: {np.degrees(extracted_yaw):.1f}°, Error: {np.degrees(yaw_error):.3f}°")
            
            # Pure roll rotation (X-axis)
            r_roll = R.from_euler('X', angle_rad)
            scipy_quat_roll = r_roll.as_quat()
            mj_quat_roll = self._scipy_quat_to_mujoco(scipy_quat_roll)
            
            extracted_yaw_roll = self._get_yaw_from_quat(mj_quat_roll)
            print(f"Roll {angle_deg}°: Extracted yaw: {np.degrees(extracted_yaw_roll):.1f}° (should be ~0)")
            
            # Pure pitch rotation (Y-axis)
            r_pitch = R.from_euler('Y', angle_rad)
            scipy_quat_pitch = r_pitch.as_quat()
            mj_quat_pitch = self._scipy_quat_to_mujoco(scipy_quat_pitch)
            
            extracted_yaw_pitch = self._get_yaw_from_quat(mj_quat_pitch)
            print(f"Pitch {angle_deg}°: Extracted yaw: {np.degrees(extracted_yaw_pitch):.1f}° (should be ~0)")
        
        print()
    
    def test_rpy_consistency(self):
        """Test RPY extraction consistency"""
        print("=== TEST 3: RPY Extraction Consistency ===")
        
        # Test various orientations
        test_cases = [
            (0, 0, 0),      # Identity
            (10, 0, 0),     # Pure roll
            (0, 10, 0),     # Pure pitch  
            (0, 0, 30),     # Pure yaw
            (10, 5, 30),    # Mixed
            (-10, -5, -30), # Mixed negative
        ]
        
        for roll_deg, pitch_deg, yaw_deg in test_cases:
            print(f"\n--- Testing RPY: ({roll_deg}, {pitch_deg}, {yaw_deg}) degrees ---")
            
            # Create rotation from RPY
            r = R.from_euler('XYZ', [np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)])
            scipy_quat = r.as_quat()
            mj_quat = self._scipy_quat_to_mujoco(scipy_quat)
            
            print(f"Original RPY: [{roll_deg}, {pitch_deg}, {yaw_deg}]")
            print(f"MuJoCo quat: {mj_quat}")
            
            # Extract using your method (ZYX convention)
            euler_zyx = R.from_quat(scipy_quat).as_euler('ZYX', degrees=True)
            rpy_extracted = np.array([euler_zyx[2], euler_zyx[1], euler_zyx[0]])  # Z,Y,X → R,P,Y
            
            print(f"Extracted RPY: [{rpy_extracted[0]:.1f}, {rpy_extracted[1]:.1f}, {rpy_extracted[2]:.1f}]")
            
            # Test yaw extraction
            yaw_extracted = self._get_yaw_from_quat(mj_quat)
            print(f"Yaw from function: {np.degrees(yaw_extracted):.1f}°")
            print(f"Yaw from RPY: {rpy_extracted[2]:.1f}°")
            
            # Check consistency
            yaw_diff = abs(np.degrees(yaw_extracted) - rpy_extracted[2])
            print(f"Yaw difference: {yaw_diff:.3f}°")
            
            if yaw_diff > 1.0:
                print("⚠️  WARNING: Large yaw difference!")
        
        print()
    
    def test_local_distance_transform(self):
        """Test local distance transformation"""
        print("=== TEST 4: Local Distance Transformation ===")
        
        # Test scenarios
        scenarios = [
            {"name": "Target in front", "drone_pos": [0, 0, 1], "target_pos": [1, 0, 1], "drone_yaw": 0},
            {"name": "Target behind", "drone_pos": [0, 0, 1], "target_pos": [-1, 0, 1], "drone_yaw": 0},
            {"name": "Target to right", "drone_pos": [0, 0, 1], "target_pos": [0, 1, 1], "drone_yaw": 0},
            {"name": "Target in front, drone rotated", "drone_pos": [0, 0, 1], "target_pos": [1, 0, 1], "drone_yaw": 90},
        ]
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            drone_pos = np.array(scenario['drone_pos'])
            target_pos = np.array(scenario['target_pos'])
            drone_yaw_deg = scenario['drone_yaw']
            
            # Create drone orientation
            r_drone = R.from_euler('Z', np.radians(drone_yaw_deg))
            scipy_quat = r_drone.as_quat()
            
            # Calculate global distance
            global_dist = target_pos - drone_pos
            
            # Transform to local frame
            rot_world_to_body = R.from_quat(scipy_quat).as_matrix()
            local_dist = rot_world_to_body @ global_dist
            
            print(f"Drone pos: {drone_pos}")
            print(f"Target pos: {target_pos}")
            print(f"Drone yaw: {drone_yaw_deg}°")
            print(f"Global distance: {global_dist}")
            print(f"Local distance: {local_dist}")
            print(f"  - Forward (x): {local_dist[0]:.2f}")
            print(f"  - Right (y): {local_dist[1]:.2f}")
            print(f"  - Up (z): {local_dist[2]:.2f}")
            
            # Sanity check
            if scenario['name'] == "Target in front" and local_dist[0] < 0:
                print("⚠️  WARNING: Target in front should have positive local x!")
            elif scenario['name'] == "Target behind" and local_dist[0] > 0:
                print("⚠️  WARNING: Target behind should have negative local x!")
    
    def test_wrap_angle(self):
        """Test angle wrapping"""
        print("=== TEST 5: Angle Wrapping ===")
        
        test_angles = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, -np.pi/2, -np.pi, -3*np.pi/2, 5*np.pi]
        
        for angle in test_angles:
            wrapped = self._wrap_angle(angle)
            print(f"Angle: {np.degrees(angle):6.1f}° → Wrapped: {np.degrees(wrapped):6.1f}°")
        
        print()
    
    def run_all_tests(self):
        """Run all tests"""
        print("🔍 TESTING TRANSFORMATIONS\n")
        
        self.test_identity_quaternion()
        self.test_pure_rotations()
        self.test_rpy_consistency()
        self.test_local_distance_transform()
        self.test_wrap_angle()
        
        print("✅ All tests completed!")

# Uruchom testy
if __name__ == "__main__":
    tester = TransformationTester()
    tester.run_all_tests()