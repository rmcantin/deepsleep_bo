# -*- coding: utf-8 -*-

import os
import re
import sys
import numpy as np
from typing import List
if sys.platform.startswith('win32'):
    from bbt_clock_win import Clock
elif sys.platform.startswith('linux'):
    from bbt_clock_linux import Clock


# PUBLIC STUFF

# clock instance for all the module
clock = Clock()


# get timestamp
def timestamp() -> int:
    """Returns the current timestamp in bbt format with microseconds resolution"""
    return clock.timestamp()


class FileInfo:
    def __init__(self, unit_information):
        self.path = unit_information['parameters']['path_main']
        self.subject = unit_information['parameters']['path_subject']
        self.session = unit_information['parameters']['path_session']
        self.file_prefix = unit_information['parameters']['file_prefix']

    def output_path(self, session=None):
        if session is None:
            session = self.session
        return os.path.join(self.path, self.subject, 'Session{}'.format(session))

    def output_filename(self, session=None, run=None):
        if session is None:
            session = self.session
        if run is None:
            run = self.__current_run()
        return r'{}S{}R{:03d}.bbt'.format(self.file_prefix, self.session, run)

    def __current_run(self):
        run_list = []
        str_regex = r'{}S{}R(\d+).bbt'.format(self.file_prefix, self.session)
        regex = re.compile(str_regex)
        for root, dirs, files in os.walk(self.path):
            for file in files:
                m = regex.match(file)
                if m:
                    run_list.append(m.group(1))
        if not run_list:
            return -1
        else:
            return int(max(run_list))


class BlockShapeError(Exception):
    """Blocks with different shape error"""
    pass


class Block:
    """
    Block is the minimum data unit for events and signals.
    Parameters:
        values: A numpy matrix of shape = (channels, samples)
        fields: A numpy matrix of shape = (2, 1). First channel is sequence number, second is battery level.
                This field is only available in blocks belonging to a signal stream.
        timestamp: The timestamp of the data. To be coherent with the units use timestamp() function above
    """
    def __init__(self, values, ts=timestamp(), fields=np.empty([0, 0])):
        self.values = values
        self.timestamp = ts
        self.fields = fields

    def to_json(self):
        output = {
            "values": self.values.tolist(),
            "timestamp": self.timestamp,
            "fields": self.fields.tolist()
        }

        return output


class Event:
    """
    An event is composed by a list of blocks
    Parameters:
        name: the name of the event as defined in the configuration files
        blocks: A list of blocks of the same shape (channels, samples) ordered chronologically
    """
    def __init__(self, name, blocks):
        self.name = name
        self.blocks = blocks
        if blocks is not None:
            if not check_blocks_shape(blocks):
                raise BlockShapeError

    def __repr__(self):
        return "Event {}: {} blocks of {} channels and {} samples".format(self.name, len(self.blocks), self.blocks[0].values.shape[0], self.blocks[0].values.shape[1])

    def to_json(self):
        """Exports the data into json format"""
        return str({
                "name": self.name,
                "blocks": [b.to_json() for b in self.blocks]
        }).replace('\'', '\"')


class Signal(Event):
    """
    A signal is composed by a list of blocks ordered chronologically
    All the blocks must be of the same shape (channels, samples)
    """
    def __init__(self, name, blocks):
        Event.__init__(self, name, blocks)

    def __repr__(self):
        return "Signal {}: {} blocks of {} channels and {} samples.".format(self.name, len(self.blocks), self.blocks[0].values.shape[0], self.blocks[0].values.shape[1])


# INTERNAL AUXILIARY FUNCTIONS
def signal_from_json(data_string: str) -> Signal:
    """Signal factory. Return a Signal object with data from json structure, called from c++ wrapper"""
    name, blocks = data_from_json(data_string)
    if name:
        return Signal(name, blocks)
    return None


def event_from_json(data_string: str) -> Event:
    """Event factory. Return an Event object with data from json structure, called from c++ wrapper"""
    name, blocks = data_from_json(data_string)
    if name:
        return Event(name, blocks)
    return None


def data_from_json(data_string: str) -> (str, List[Block]):
    """Parse a json string looking for a name and a list of blocks"""
    def block_from_data(data):
        try:
            if isinstance(data, dict) and "values" in data and "timestamp" in data:
                values = np.asarray(data["values"])
                timestamp = data["timestamp"]
                if "fields" in data:
                    return Block(values, timestamp, np.asarray(data["fields"]))
                else:
                    return Block(values, timestamp)
        except:
            print("Exception importing block from data")
            return None

    try:
        raw = data_string
        if isinstance(raw, dict) and "name" in raw and "blocks" in raw:
            return raw["name"], list(map(block_from_data, raw["blocks"]))
        else:
            return None, None  # no name or blocks in json dict
    except Exception as e:
        print("Exception importing data from json {}".format(e))
        return None, None


def check_blocks_shape(blocks) -> bool:
    """Check if all the blocks have the same shape"""
    if len(blocks) == 0:
        return True

    shape = blocks[0].values.shape
    return all(b.values.shape == shape for b in blocks)


# TEST and example
if __name__ == '__main__':

    signal_json = {"name": "EEG", "blocks": [{"values": [[1,2],[3,4],[5,6]], "timestamp": 1234, "fields": [21,43]},{"values": [[-1,-2],[-3,-4],[-5,-6]], "timestamp": 1235, "fields": [12,34]}]}

    print("get signal from " + str(signal_json))
    s = signal_from_json(signal_json)
    print(s)
    if s is not None:
        print(s.to_json())

    event_json = {"name": "Stimulus", "blocks": [{"values": [[2]], "timestamp": 1235},{"values": [[3]], "timestamp": 1236}]}
    print("get event from " + str(event_json))
    e = event_from_json(event_json)
    print(e)
    if e is not None:
        print(e.to_json())
