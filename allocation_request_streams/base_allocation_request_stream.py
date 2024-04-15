class BaseAllocationRequestStream:
    def __init__(self, allocation_request_stream_name: str):
        self.allocation_request_stream_name = allocation_request_stream_name

    def __str__(self):
        return f"AllocationRequestStream: {self.allocation_request_stream_name}"
    
    def get_next_req():
        """
        return the next allocation request
        """
        raise NotImplementedError()
    
    def allocation_request_distribution(self):
        """
        return the distribution of allocation requests
        """
        raise NotImplementedError()
    
    def allocation_request_trajectory(self):
        """
        sample a trajectory of allocation requests
        """
        raise NotImplementedError()
