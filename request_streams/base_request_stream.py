class BaseRequestStream:
    def __init__(self, allocation_request_stream_name: str):
        self.allocation_request_stream_name = allocation_request_stream_name
        self.allocated_indexes = []

    def __str__(self):
        return f"AllocationRequestStream: {self.allocation_request_stream_name}"
    
    def get_next_req():
        """
        return the next allocation request

        output:
            tuple(free_or_alloc: {0,1}, mem_addr_or_amt: int)
        """
        raise NotImplementedError()
    

    def sample_request_distribution(self):
            """
            return the distribution of allocation requests
            """
            raise NotImplementedError()