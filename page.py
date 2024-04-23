import numpy as np


class Page():
    def __init__(self, page_size=256, block_size=1) -> None:
        self.page_size = page_size
        self.block_size = block_size
        self.bitmap = np.zeros(page_size, dtype=np.uint8)
        self.free_list = [{"idx": 0, 
                           "size": page_size}
                        ]
        self.allocated_list = {}
    def _update_free_list(self):
        is_zero = np.concatenate(([True], self.bitmap == 1, [True]))
        abs_diff = np.abs(np.diff(is_zero.astype(int)))

        ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)

        self.free_list =  [{"idx": start, "size": end-start} for start, end in ranges]

    def allocate(self, address, amount):
        if address < 0 or address + amount > self.page_size:
            return False
        if np.all(self.bitmap[address:address + amount] == 0):
            self.bitmap[address:address + amount] = 1
            self.allocated_list[address] = amount
            self._update_free_list()
            return True
        return False

    def free(self, address):
        
        assert address in self.allocated_list , f"Address {address} was never allocated!"
        amt_to_free = self.allocated_list[address]
        del self.allocated_list[address]
        assert np.all(self.bitmap[address:address + amt_to_free] == 1), "Bitmap and allocated list not consistent"
        self.bitmap[address:address + amt_to_free] = 0
        self._update_free_list()

    def space_available(self, amount):
        for free_block in self.free_list:
            if free_block["size"] >= amount:
                return True
        
        return False


