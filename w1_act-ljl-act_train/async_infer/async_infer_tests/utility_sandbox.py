from async_infer.gather_obs_typedef import SingleReaderGatherDataInterfaceAsyncQueue


def _test_single_read_gather_data_simple():
    fifo = SingleReaderGatherDataInterfaceAsyncQueue[str](rough_max_size=4)
    fifo.append('a', 0.0)
    fifo.append('b', 0.1)
    out = fifo(request_time=0.1)
    assert len(out) == 2
    for i in range(100):
        fifo.append(f'c_{i}', float(i))
    assert fifo.rough_size() < 10
    out = fifo(request_time=0.2)
    assert len(out) >= 3


def _test_single_read_gather_data_concurrent():
    import threading
    import time

    # Create queue with large enough size to hold all items
    fifo = SingleReaderGatherDataInterfaceAsyncQueue[str](rough_max_size=100)

    # List to collect all read items
    collected_items = []

    # Number of writers and items per writer
    num_writers = 3
    items_per_writer = 20

    def writer_function(writer_id):
        """Writer thread function"""
        for i in range(items_per_writer):
            item = f'writer{writer_id}_item{i}'
            # Append with a timestamp (using current time for realism)
            fifo.append(item, time.time())
            # Small delay to simulate work
            time.sleep(0.01)

    # Create and start writer threads
    writer_threads = []
    for i in range(num_writers):
        thread = threading.Thread(target=writer_function, args=(i,))
        writer_threads.append(thread)
        thread.start()

    # Read from main thread while writers are working
    for _ in range(20):  # Read multiple times
        items = fifo(request_time=time.time())
        collected_items.extend(item.data for item in items)
        # Small delay between reads
        time.sleep(0.05)

    # Wait for all writer threads to finish
    for thread in writer_threads:
        thread.join()

    # Do a final read to get any remaining items
    final_items = fifo(request_time=time.time())
    collected_items.extend(item.data for item in final_items)

    # Verify all items were collected
    expected_items = num_writers * items_per_writer
    assert len(collected_items) == expected_items, f"Expected {expected_items} items, got {len(collected_items)}"

    # Verify each writer's items are present
    for writer_id in range(num_writers):
        writer_items = [item for item in collected_items if f'writer{writer_id}_' in item]
        assert len(
            writer_items) == items_per_writer, f"Writer {writer_id} should have {items_per_writer} items, got {len(writer_items)}"

    print(f"Concurrent test passed! Collected {len(collected_items)} items from {num_writers} writers.")


if __name__ == '__main__':
    _test_single_read_gather_data_simple()
    _test_single_read_gather_data_concurrent()
