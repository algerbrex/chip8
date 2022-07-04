"""
An implementation of a Chip-8 emulator. The emulator supports most
common chip8 programs such as pong or breakout.

Copyright (c) 2022 Christian M. Dean
"""


from sys import argv
from typing import Generator, Sequence
from collections import namedtuple
from random import randint
from enum import IntEnum
from numpy import byte

import pygame as pg

pg.init()
pg.mixer.init()

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 320
FPS = 120
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RAM_SIZE = 4096
NUM_REGISTERS = 16
STACK_SIZE = 16
PROGRAM_START_ADDR = 0x200
CHIP8_ENDIANESS = 'big'
PIXEL_SIZE = 10 # 1 chip-8 pixel = 10 regular pixels, to increase resolution

TIMER_DECREMENT_RATE = 16 # approx. 60Hz
UPDATE_TIMERS_EVENT = pg.USEREVENT + 1

HEXADECIMAL_KEYPAD_TO_KEYS = {
    0x1: pg.K_1,
    0x2: pg.K_2,
    0x3: pg.K_3,
    0xC: pg.K_4,

    0x4: pg.K_q,
    0x5: pg.K_w,
    0x6: pg.K_e,
    0xD: pg.K_r,

    0x7: pg.K_a,
    0x8: pg.K_s,
    0x9: pg.K_d,
    0xE: pg.K_f,

    0xA: pg.K_z,
    0x0: pg.K_x,
    0xB: pg.K_c,
    0xF: pg.K_v,
}

KEYS_TO_HEXADECIMAL_KEYPAD = {v: k for k, v in HEXADECIMAL_KEYPAD_TO_KEYS.items()}

HEXADECIMAL_SPRITES = [
    bytearray((0xF0, 0x90, 0x90, 0x90, 0xF0)),
    bytearray((0x20, 0x60, 0x20, 0x20, 0x70)),
    bytearray((0xF0, 0x10, 0xF0, 0x80, 0xF0)),
    bytearray((0xF0, 0x10, 0xF0, 0x10, 0xF0)),

    bytearray((0x90, 0x90, 0xF0, 0x10, 0x10)),
    bytearray((0xF0, 0x80, 0xF0, 0x10, 0xF0)),
    bytearray((0xF0, 0x80, 0xF0, 0x90, 0xF0)),
    bytearray((0xF0, 0x10, 0x20, 0x40, 0x40)),

    bytearray((0xF0, 0x90, 0xF0, 0x90, 0xF0)),
    bytearray((0xF0, 0x90, 0xF0, 0x10, 0xF0)),
    bytearray((0xF0, 0x90, 0xF0, 0x90, 0x90)),
    bytearray((0xE0, 0x90, 0xE0, 0x90, 0xE0)),

    bytearray((0xF0, 0x80, 0x80, 0x80, 0xF0)),
    bytearray((0xE0, 0x90, 0x90, 0x90, 0xE0)),
    bytearray((0xF0, 0x80, 0xF0, 0x80, 0xF0)),
    bytearray((0xF0, 0x80, 0xF0, 0x80, 0x80))
]

HEXADECIMAL_SPRITES_START_ADDRS = {
    0x0: 0x0,
    0x1: 0x5,
    0x2: 0xA,
    0x3: 0xF,
    0x4: 0x14,
    0x5: 0x19,
    0x6: 0x1E,
    0x7: 0x23,
    0x8: 0x28,
    0x9: 0x2D,
    0xA: 0x32,
    0xB: 0x37,
    0xC: 0x3C,
    0xD: 0x41,
    0xE: 0x46,
    0xF: 0x4B
}

Instruction = namedtuple('Instruction', 'opcode nnn n x y nn all')

class State(IntEnum):
    NO_IO                  = 0
    CHECK_KEYS_PRESSED     = 1
    PAUSE_UNTIL_KEY_PRESS  = 2

class VirtualMachineError(Exception):
    pass


class ColoredRect(pg.Rect):
    def __init__(self, color, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.color = color

class VirtualDisplay:
    def __init__(self) -> None:
        self._pixel_coords: dict[(int, int), ColoredRect] = {}

        for y_coord in range(0, WINDOW_HEIGHT, PIXEL_SIZE):
            for x_coord in range(0, WINDOW_WIDTH, PIXEL_SIZE):
                pixel: ColoredRect = ColoredRect(BLACK, x_coord, y_coord, PIXEL_SIZE, PIXEL_SIZE)
                self._pixel_coords[(x_coord, y_coord)] = pixel

    def get_pixel_at(self, x_coord: int, y_coord: int) -> ColoredRect:
        pixel = self._pixel_coords.get((x_coord, y_coord))
        if pixel is None:
            raise VirtualMachineError("Atempt to access bad coordinates on virtual display:", (x_coord, y_coord))
        return pixel

    def clear(self):
        for pixel in self._pixel_coords.values():
            pixel.color = BLACK

    def render(self, surface: pg.Surface):
        for pixel in self._pixel_coords.values():
            pg.draw.rect(surface, pixel.color, pixel)


class VirtualMachine:
    def __init__(self) -> None:
        self._ram: bytearray = bytearray(RAM_SIZE)
        self._registers: bytearray = bytearray(NUM_REGISTERS)
        self._stack: list[int] = [0] * STACK_SIZE

        self._I_register: int
        self._delay_timer_register: int
        self._sound_timer_register: int

        self._program_counter: int
        self._stack_pointer: int

        self.virtual_display = VirtualDisplay()

    def load(self, program: bytearray) -> None:
        self._ram: bytearray = bytearray(RAM_SIZE)

        for i, sprite in enumerate(HEXADECIMAL_SPRITES):
            self._ram[i*5:i*5+5] = sprite[:]

        for index, addr in enumerate(range(PROGRAM_START_ADDR, RAM_SIZE)):
            if index == len(program):
                break

            self._ram[addr] = program[index]

        self._registers = bytearray(NUM_REGISTERS)
        self._stack = [0] * STACK_SIZE

        self._I_register = 0
        self._delay_timer_register = 0
        self._sound_timer_register = 0

        self._program_counter = PROGRAM_START_ADDR
        self._stack_pointer = -1

    def execute(self, display: pg.Surface):
        while True:
            if self._program_counter >= RAM_SIZE:
                raise VirtualMachineError("Program counter overflowing memory addresses")
            
            raw_instruction = int.from_bytes(self._ram[self._program_counter:self._program_counter+2], CHIP8_ENDIANESS)
            instruction = Instruction (
                (raw_instruction & 0xf000) >> 12,
                raw_instruction & 0xfff,
                raw_instruction & 0xf,
                (raw_instruction & 0xf00) >> 8,
                (raw_instruction & 0xf0) >> 4,
                raw_instruction & 0xff,
                raw_instruction
            )

            increment_pc: bool = True

            if instruction.all == 0x00E0:
                # 0x00E0 -> clear the display
                self.virtual_display.clear()
                yield State.NO_IO
            elif instruction.all == 0x00EE:
                # 0x00EE -> return from a subroutine
                self._program_counter = self._stack[self._stack_pointer]
                self._stack_pointer -= 1
                yield State.NO_IO
            elif instruction.opcode == 0x1:
                # 0x1nnn -> jump to the address nnn
                self._program_counter = instruction.nnn
                increment_pc = False
                yield State.NO_IO
            elif instruction.opcode == 0x2:
                # 0x2nnn -> call a subroutine at the address nnn
                self._stack_pointer += 1
                self._stack[self._stack_pointer] = self._program_counter
                self._program_counter = instruction.nnn
                increment_pc = False
                yield State.NO_IO
            elif instruction.opcode == 0x3:
                # 0x3xnn -> skip an instruction given Vx == nn
                if self._registers[instruction.x] == instruction.nn:
                    self._program_counter += 2
                yield State.NO_IO
            elif instruction.opcode == 0x4:
                # 0x4xnn -> skip an instruction given Vx != nn
                if self._registers[instruction.x] != instruction.nn:
                    self._program_counter += 2
                yield State.NO_IO
            elif instruction.opcode == 0x5:
                # 0x5xy0 -> skip an instruction given Vx == Vy
                if self._registers[instruction.x] == self._registers[instruction.y]:
                    self._program_counter += 2
                yield State.NO_IO
            elif instruction.opcode == 0x6:
                # 0x6xnn -> store nn in Vx
                self._registers[instruction.x] = instruction.nn
                yield State.NO_IO
            elif instruction.opcode == 0x7:
                # 0x7xnn -> store the value of Vx + nn in Vx, only getting the lower 8 bits if the result overflows 8-bits.
                self._registers[instruction.x] = (self._registers[instruction.x] + instruction.nn) & 0xff
                yield State.NO_IO
            elif instruction.opcode == 0x8:
                if instruction.n == 0x0:
                    # 0x8xy0 -> store the value of Vy in Vx
                    self._registers[instruction.x] = self._registers[instruction.y]
                elif instruction.n == 0x1:
                    # 0x8xy1 -> store the value of Vx | Vy in Vx
                    self._registers[instruction.x] = self._registers[instruction.x] | self._registers[instruction.y]
                elif instruction.n == 0x2:
                    # 0x8xy2 -> store the value of Vx & Vy in Vx
                    self._registers[instruction.x] = self._registers[instruction.x] & self._registers[instruction.y]
                elif instruction.n == 0x3:
                    # 0x8xy3 -> store the value of Vx ^ Vy in Vx
                    self._registers[instruction.x] = self._registers[instruction.x] ^ self._registers[instruction.y]
                elif instruction.n == 0x4:
                    # 0x8xy4 -> store the value of Vx + Vy in Vx and set VF to 01 if a carry occurs otherwise 00
                    # if the sum does overflow, only store the lower 8 bits.
                    result = self._registers[instruction.x] + self._registers[instruction.y]
                    self._registers[0xF] = 1 if result > 255 else 0
                    self._registers[instruction.x] = result & 0xff
                elif instruction.n == 0x5:
                    # 0x8xy5 -> store the value of Vx - Vy in Vx and set VF to 01 if Vx > Vy, otherwise
                    # set VF to 00, denoting a borrow has occured.
                    result = self._registers[instruction.x] - self._registers[instruction.y]
                    self._registers[0xF] = 1 if result > 0 else 0
                    self._registers[instruction.x] = result & 0xff
                elif instruction.n == 0x6:
                    # 0x8xy6 -> store Vy >> 1 in Vx and set VF to the least singificant bit
                    # of Vx prior to the shift.
                    self._registers[0xF] = self._registers[instruction.x] & 0x1
                    self._registers[instruction.x] = self._registers[instruction.y] >> 1
                elif instruction.n == 0x7:
                    # 0x8xy7 -> store Vy - Vx in Vx, and set VF to 01 if Vy > Vx, else 00,
                    # denoting a borrow has occured.
                    result = self._registers[instruction.y] - self._registers[instruction.x]
                    self._registers[0xF] = 1 if result > 0 else 0
                    self._registers[instruction.x] = result & 0xff
                elif instruction.n == 0xE:
                    # 0x8xyE -> store Vy << 1 in Vx and set VF to the most singificant bit
                    # of Vx prior to the shift.
                    self._registers[0xF] = (self._registers[instruction.x] & 0x80) >> 7
                    self._registers[instruction.x] = (self._registers[instruction.y] << 1) & 0xff
                else:
                    raise VirtualMachineError(f'Unrecognized instruction {instruction} at memory address {self._program_counter}')
                yield State.NO_IO
            elif instruction.opcode == 0x9:
                # 9xy0 -> skip an instruction if Vx != Vy
                if self._registers[instruction.x] != self._registers[instruction.y]:
                    self._program_counter += 2
                yield State.NO_IO
            elif instruction.opcode == 0xA:
                # Annn -> store memory address nnn in I
                self._I_register = instruction.nnn
                yield State.NO_IO
            elif instruction.opcode == 0xB:
                # Bnnn -> jump to address nnn + V0
                self._program_counter = instruction.nnn + self._registers[0x0]
                increment_pc = False
                yield State.NO_IO
            elif instruction.opcode == 0xC:
                # Cxnn -> set Vx to a random number between [0, 255], bitwise ANDed with nn
                self._registers[instruction.x] = randint(0, 255) & instruction.nn
            elif instruction.opcode == 0xD:
                # Dxyn -> Display an n-byte sprite starting at I at (Vx, Vy), and set
                # VF to 01 if a collision with an exisiting sprite on screen occurs.

                self._registers[0xF] = 0x0

                sprite: bytearray = bytearray(instruction.n)
                for i in range(instruction.n):
                    sprite[i] = self._ram[self._I_register+i]

                x_start, y_start = self._registers[instruction.x] * PIXEL_SIZE, self._registers[instruction.y] * PIXEL_SIZE
                x_coord, y_coord = x_start, y_start

                for byte in sprite:
                    x_coord = x_start
                    for bit in map(int, format(byte, '08b')):
                        rect: ColoredRect = self.virtual_display.get_pixel_at(x_coord % WINDOW_WIDTH, y_coord % WINDOW_HEIGHT)
                        curr_color: Sequence[int, int, int] = display.get_at((rect.x, rect.y))[:3]

                        if bit == 1:
                            if curr_color == WHITE:
                                self._registers[0xF] = 0x1

                            rect.color = WHITE if curr_color == BLACK else BLACK
                        else:
                            rect.color = WHITE if curr_color == WHITE else BLACK

                        x_coord += PIXEL_SIZE
                    y_coord += PIXEL_SIZE

                yield State.NO_IO
            elif instruction.opcode == 0xE:
                yield State.CHECK_KEYS_PRESSED
                key_states: Sequence[bool] = yield

                if instruction.nn == 0x9E:
                    # Ex9E -> skip an instruction if the key corresponding to the value of Vx is pressed.
                    if key_states[HEXADECIMAL_KEYPAD_TO_KEYS[self._registers[instruction.x]]]:
                        self._program_counter += 2
                elif instruction.nn == 0xA1:
                    # ExA1 -> skip an instruction if the key corresponding to the value of Vx is not pressed.
                    if not key_states[HEXADECIMAL_KEYPAD_TO_KEYS[self._registers[instruction.x]]]:
                        self._program_counter += 2
                else:
                    raise VirtualMachineError(f'Unrecognized instruction {instruction} at memory address {self._program_counter}')
                yield
            elif instruction.opcode == 0xF:
                if instruction.nn == 0x07:
                    # Fx07 -> set Vx to be delay timer value.
                    self._registers[instruction.x] = self._delay_timer_register
                    yield State.NO_IO
                elif instruction.nn == 0x0A:
                    # Fx0A -> pause execution until key press is recived
                    yield State.PAUSE_UNTIL_KEY_PRESS
                    key: int = yield
                    self._registers[instruction.x] = key
                    yield
                elif instruction.nn == 0x15:
                    # Fx15 -> store the value of Vx in the delay timer.
                    self._delay_timer_register = self._registers[instruction.x]
                    yield State.NO_IO
                elif instruction.nn == 0x18:
                    # Fx18 -> store the value of Vx in the sounder timer.
                    self._sound_timer_register = self._registers[instruction.x]
                    yield State.NO_IO
                elif instruction.nn == 0x1E:
                    # Fx1E -> Store the value of I + Vx into I.
                    self._I_register = self._I_register + self._registers[instruction.x]
                    yield State.NO_IO
                elif instruction.nn == 0x29:
                    # Fx29 -> set I = location of sprite correpsonding to hexadecimal digit inside of Vx.
                    self._I_register = HEXADECIMAL_SPRITES_START_ADDRS[self._registers[instruction.x]]                    
                    yield State.NO_IO
                elif instruction.nn == 0x33:
                    # Fx33 -> store the BCD representaion of Vx in memory locations I, I+1, and I+2.
                    hundreds_digit, tens_digit, ones_digit = self.get_digits_of_num(self._registers[instruction.x])
                    self._ram[self._I_register] = hundreds_digit
                    self._ram[self._I_register + 1] = tens_digit
                    self._ram[self._I_register + 2] = ones_digit
                    yield State.NO_IO
                elif instruction.nn == 0x55:
                    # Fx55 -> Copy registers V0 through Vx into RAM, starting at the address stored in I.
                    idx = 0
                    while idx <= instruction.x:
                        self._ram[self._I_register+idx] = self._registers[idx]
                        idx += 1
                    yield State.NO_IO
                elif instruction.nn == 0x65:
                     # Fx65 -> Read registers V0 through Vx from RAM, starting at the address stored in I.
                    idx = 0
                    while idx <= instruction.x:
                        self._registers[idx] = self._ram[self._I_register+idx]
                        idx += 1
                    yield State.NO_IO
                else:
                    raise VirtualMachineError(f'Unrecognized instruction {instruction} at memory address {self._program_counter}')
            else:
                if instruction.all == 0x0:
                    yield State.NO_IO
                    continue
                raise VirtualMachineError(f'Unrecognized instruction {instruction} at memory address {self._program_counter}')

            # For some instructions, such as 0xA, or 0x1, the program counter is already set to the exact correct
            # address, and incrementing the program counter again would cause instructions to be skipped incorrectly,
            # so make sure to only increment the program counter for the appropriate instructions.
            if increment_pc: 
                self._program_counter += 2

    def should_play_buzzer(self) -> bool:
        return self._sound_timer_register > 0

    def update_timers(self) -> None:
        if self._delay_timer_register > 0:
            self._delay_timer_register -= 1
        
        if self._sound_timer_register > 0:
            self._sound_timer_register -=1

    @staticmethod
    def get_digits_of_num(num: int) -> list[int]:
        digits, idx = [0, 0, 0], 2
        while num != 0:
            digits[idx] = num % 10
            num //= 10
            idx -= 1
        return digits


if __name__ == '__main__':
    vm: VirtualMachine = VirtualMachine()
    with open(argv[1], 'rb') as infile:
        program_binary: byte = bytearray(infile.read())
        vm.load(program_binary)

    display = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pg.time.Clock()
    buzzer = pg.mixer.Sound("buzzer.wav")
    pg.time.set_timer(UPDATE_TIMERS_EVENT, TIMER_DECREMENT_RATE)

    gen: Generator = vm.execute(display)
    buzzer_playing: bool = False

    while True:
        if vm.should_play_buzzer() and not buzzer_playing:
            buzzer.play()
            buzzer_playing = True
        
        if not vm.should_play_buzzer() and buzzer_playing:
            buzzer.stop()
            buzzer_playing = False

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
            elif event.type == UPDATE_TIMERS_EVENT:
                vm.update_timers()

        state: State = next(gen)

        if state == State.NO_IO:
            pass
        elif state == State.CHECK_KEYS_PRESSED:
            next(gen)
            gen.send(pg.key.get_pressed())
        elif state == State.PAUSE_UNTIL_KEY_PRESS:
            paused = True
            while paused:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        quit()
                    elif event.type == pg.KEYDOWN and event.key in KEYS_TO_HEXADECIMAL_KEYPAD:
                        next(gen)
                        gen.send(KEYS_TO_HEXADECIMAL_KEYPAD[event.key])
                        paused = False
                        break
        else:
            raise VirtualMachineError('Unrecognized VM state:', state)

        display.fill(BLACK)
        vm.virtual_display.render(display)
        pg.display.flip()
        clock.tick(FPS)
