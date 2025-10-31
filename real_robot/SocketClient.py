import zmq
import numpy as np
import time

class FrankaClient:
    """
    Though we can send numpy object, this is 10 times slower than sending a list since it will contain all definations of numpy!
    """
    def __init__(self, controller_ip='152.3.172.50', sub_port=5555, pub_port=5556):
        self.context = zmq.Context()
        
        # State subscriber
        self.state_sub = self.context.socket(zmq.SUB)
        self.state_sub.setsockopt(zmq.CONFLATE, 1) # Only keep latest message no queuing
        self.state_sub.connect(f"tcp://{controller_ip}:{sub_port}")
        self.state_sub.setsockopt_string(zmq.SUBSCRIBE, "") # No identity filter, accept all messages
        
        # Command publisher
        self.cmd_pub = self.context.socket(zmq.PUB)
        self.cmd_pub.connect(f"tcp://{controller_ip}:{pub_port}")


    def get_state(self):
        """Get latest robot state (non-blocking)"""
        try:
            return self.state_sub.recv_pyobj(zmq.NOBLOCK)
        except zmq.Again:
            return None

    def send_command(self, value, cmd="osc"):
        """Send control command to robot"""
        cmd_dict = {
            'type': cmd,
            'value': value
        }
        self.cmd_pub.send_pyobj(cmd_dict)

    def run_policy(self, policy_fn):
        """Run custom policy function at 100Hz"""
        while True:
            start_time = time.time()
            state = self.get_state()
            if state:
                value = policy_fn(state)
                self.send_command(value)
                print(f"Loop time: {time.time() - start_time}")


    def stop(self):
        self.state_sub.close()
        self.cmd_pub.close()
        self.context.term()
        

# Example usage
if __name__ == "__main__":
    client = FrankaClient()
    
    def simple_policy(state):
        return np.random.uniform(-0.1, 0.1, 7)
    
    client.run_policy(simple_policy)