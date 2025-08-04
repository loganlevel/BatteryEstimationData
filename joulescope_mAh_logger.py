#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Jetperch LLC
# Licensed under the Apache License, Version 2.0

import argparse
import threading
import time
import csv
import os
from joulescope import scan


def get_parser():
    p = argparse.ArgumentParser(description='Log charge and energy every X minutes.')
    p.add_argument('-i', '--interval', type=float, default=5, help='Logging interval in minutes (default: 5)')
    p.add_argument('-o', '--output', type=str, default='battery_log.csv', help='CSV file to save output (default: battery_log.csv)')
    return p


class BatteryLogger:

    def __init__(self, interval_minutes, csv_filename):
        self.interval = interval_minutes * 60  # Convert to seconds
        self.csv_filename = csv_filename
        self._stat_first = None
        self._stat_now = None
        self._reset_interval_data()

        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'elapsed_time', 'mAh',
                    'avg_voltage_V', 'min_voltage_V', 'avg_current_A', 'max_current_A'
                ])

    def _reset_interval_data(self):
        self.interval_start_time = time.time()
        self.voltages = []
        self.currents = []

    def statistics_callback(self, stats):
        t = stats['time']['range']['value'][0]
        v = stats['signals']['voltage']['µ']['value']
        i = stats['signals']['current']['µ']['value']

        if self._stat_first is None:
            self._stat_first = stats
            self._logging_start_time = time.time()
        self._stat_now = stats

        elapsed = (time.time() - self._logging_start_time) / 60  # elapsed time in minutes
        mAh = (stats['accumulators']['charge']['value'] - self._stat_first['accumulators']['charge']['value']) / 3.6

        self.voltages.append(v)
        self.currents.append(i)

        now = time.time()
        if now - self.interval_start_time >= self.interval:
            avg_v = sum(self.voltages) / len(self.voltages)
            min_v = min(self.voltages)
            avg_i = sum(self.currents) / len(self.currents)
            max_i = max(self.currents)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, round(elapsed, 1), round(mAh, 9),
                    round(avg_v, 3), round(min_v, 3), round(avg_i, 9), round(max_i, 9)
                ])

            print(f'[{timestamp}] Log written, {round(mAh, 3)} mAh')
            self._reset_interval_data()


def run():
    args = get_parser().parse_args()
    devices = scan(config='off')
    if not devices:
        print('No Joulescope device found')
        return 1

    device = devices[0]
    logger = BatteryLogger(args.interval, args.output)
    device.statistics_callback_register(logger.statistics_callback, 'sensor')
    device.open()

    try:
        device.parameter_set('i_range', 'auto')
        device.parameter_set('v_range', '15V')
        print("Logging started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Interrupted by user.')
    finally:
        device.parameter_set('i_range', 'off')
        device.close()


if __name__ == '__main__':
    run()
