import os
import sys
import pygame

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ConnectFourEnv(gym.Env):
    def __init__(self, render_mode = None):
        super().__init__()
        self.rows = 6
        self.cols = 7
        self.connect = 4
        self.action_space = spaces.Discrete(self.cols)
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (self.rows, self.cols), dtype = np.float16)
        
        self.render_mode = render_mode  # 'human' for GUI, 'text' for terminal, None for no render
        self.pygame = None
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.number_font = None
        
        if self.render_mode == 'human':
            self._init_pygame()
        
        self.reset()
    
    def reset(self):
        self.done = False
        self.winner = None
        self.current_player = 1
        self.board = np.zeros((self.rows, self.cols), dtype = np.float16)

        return self.board.copy()
    
    def step(self, action):
        if not self.is_valid_action(action):
            return self.board.copy(), -1, True, {"invalid_move": True}
        
        row = self.get_next_open_row(action)
        
        # Animation for GUI mode
        if self.render_mode == 'human' and self.pygame is not None:
            self._gui_animation(row, action)
        
        # Place the piece on the board
        self.board[row, action] = self.current_player
        
        if self.check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), 1, True, {"winner": self.current_player}
        
        if self.is_board_full():
            self.done = True
            return self.board.copy(), 0, True, {"draw": True}
        
        reward = self.get_heuristic_reward(row, action)
        
        self.current_player *= -1
        return self.board.copy(), reward, False, {}
    
    def is_valid_action(self, col):
        return col >= 0 and col < self.cols and self.board[0, col] == 0
    
    def get_next_open_row(self, col):
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        return -1
    
    def check_win(self, player):
        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row, col + i] == player for i in range(4)):
                    return True
        
        # Vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(self.board[row + i, col] == player for i in range(4)):
                    return True
        
        # Diagonal
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return True
        
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row - i, col + i] == player for i in range(4)):
                    return True
        
        return False
    
    def is_board_full(self):
        return np.all(self.board != 0)
    
    def get_valid_actions(self):
        return [col for col in range(self.cols) if self.is_valid_action(col)]
    
    def get_heuristic_reward(self, action_row, action_col):
        reward = 0.0
        player = self.current_player

        # Check if the play can win by making a different actions
        self.board[action_row][action_col] = 0

        for valid_action in self.get_valid_actions():
            if valid_action == action_col:
                continue

            row = self.get_next_open_row(valid_action)
            if row == -1: # Column full
                continue

            self.board[row][valid_action] = player

            if self.check_win(player):
                reward -= 0.5
            
            self.board[row][valid_action] = 0

        self.board[action_row][action_col] = player
        
        # Check if opponent can win on their next turn
        for opponent_action in self.get_valid_actions():
            row = self.get_next_open_row(opponent_action)
            if row == -1:  # Column full
                continue
                
            # Temporarily place opponent's piece
            self.board[row][opponent_action] = -player
            
            if self.check_win(-player):
                reward -= 0.5
            
            self.board[row][opponent_action] = 0
        
        # Check if this move blocked opponent's winning move
        blocking_bonus = self._check_if_blocked_win(action_row, action_col, player)
        reward += blocking_bonus
        
        return reward

    def _check_if_blocked_win(self, action_row, action_col, player):
        self.board[action_row][action_col] = -player        
        blocked_win = self.check_win(-player)
        self.board[action_row][action_col] = player
        
        return 0.15 if blocked_win else 0.0
    
    def render(self):
        if self.render_mode == 'human':
            self._render_gui()
        elif self.render_mode == 'text':
            self._render_text()
    
    def _render_text(self):
        """Text-based rendering"""
        symbols = {0: '·', 1: 'X', -1: 'O'}
        print("\n " + " ".join(str(i) for i in range(self.cols)))
        print(" " + "-" * (self.cols * 2 - 1))
        for row in self.board:
            print(" " + " ".join(symbols[int(val)] for val in row))
        print()
        
        if self.done:
            if self.winner:
                print(f"Player {'X' if self.winner == 1 else 'O'} wins!")
            else:
                print("It's a draw!")
    
    def _init_pygame(self):
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
        
        pygame.init()
        self.pygame = pygame
        
        self.top_bar_height = 50
        self.bottom_bar_height = 50
        self.square_size = 90
        self.width = self.cols * self.square_size
        self.height = self.top_bar_height + (self.rows * self.square_size) + self.bottom_bar_height
        self.radius = int(self.square_size / 2 - 5)
        
        # Colors - using class constants
        self.BLUE = (41, 128, 185)
        self.DARK_BLUE = (30, 90, 140)
        self.BLACK = (20, 20, 30)
        self.RED = (231, 76, 60)
        self.YELLOW = (241, 196, 15)
        self.WHITE = (245, 245, 245)
        self.GRAY = (149, 165, 166)
        self.RED_HIGHLIGHT = (255, 120, 100)
        self.YELLOW_HIGHLIGHT = (255, 220, 100)
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Connect Four')
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 48, bold = True)
        self.small_font = pygame.font.SysFont("Arial", 32)
        self.number_font = pygame.font.SysFont("Arial", 36, bold = True)
        
        # Gradient for top bar
        self._gradient_cache = [
            tuple(int(self.DARK_BLUE[j] + (self.BLUE[j] - self.DARK_BLUE[j]) * (i / self.square_size)) 
                for j in range(3))
            for i in range(self.square_size)
        ]

    def _handle_events(self):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
                sys.exit(0)

    def _draw_gradient_bar(self):
        for i, color in enumerate(self._gradient_cache):
            self.pygame.draw.rect(self.screen, color, (0, i, self.width, 1))

    def _draw_piece(self, center_x, center_y, player):
        # Shadow
        self.pygame.draw.circle(self.screen, self.DARK_BLUE, 
                            (center_x + 2, center_y + 2), self.radius)
        
        if player == 0:
            self.pygame.draw.circle(self.screen, self.BLACK, 
                                (center_x, center_y), self.radius)
            
        elif player == 1:
            self.pygame.draw.circle(self.screen, self.RED, 
                                (center_x, center_y), self.radius)
            self.pygame.draw.circle(self.screen, self.RED_HIGHLIGHT, 
                                (center_x - 8, center_y - 8), self.radius // 4)
            
        else:
            self.pygame.draw.circle(self.screen, self.YELLOW, 
                                (center_x, center_y), self.radius)
            self.pygame.draw.circle(self.screen, self.YELLOW_HIGHLIGHT, 
                                (center_x - 8, center_y - 8), self.radius // 4)

    def _draw_board_grid(self):
        for c in range(self.cols):
            for r in range(self.rows):
                # Draw blue square
                self.pygame.draw.rect(self.screen, self.BLUE, 
                                    (c * self.square_size, self.top_bar_height + r * self.square_size, 
                                    self.square_size, self.square_size))
                
                center_x = int(c * self.square_size + self.square_size / 2)
                center_y = int(self.top_bar_height + r * self.square_size + self.square_size / 2)
                
                self._draw_piece(center_x, center_y, self.board[r][c])

    def _draw_column_numbers(self):
        bottom_y = self.top_bar_height + (self.rows * self.square_size) + (self.bottom_bar_height // 2)
        for c in range(self.cols):
            num_text = self.number_font.render(str(c), True, self.WHITE)
            text_rect = num_text.get_rect(
                center = (c * self.square_size + self.square_size // 2, bottom_y)
            )
            self.screen.blit(num_text, text_rect)

    def _draw_turn_indicator(self):
        if not self.done:
            player_text = "RED's Turn" if self.current_player == 1 else "YELLOW's Turn"
            player_color = self.RED if self.current_player == 1 else self.YELLOW
            turn_label = self.small_font.render(player_text, True, player_color)
            turn_rect = turn_label.get_rect(center=(self.width // 2, 30))
            self.screen.blit(turn_label, turn_rect)

    def _draw_game_over(self):
        overlay = self.pygame.Surface((self.width, self.height), self.pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        if self.winner:
            text = "RED WINS!" if self.winner == 1 else "YELLOW WINS!"
            color = self.RED if self.winner == 1 else self.YELLOW
        else:
            text = "DRAW!"
            color = self.GRAY
        
        label = self.font.render(text, True, color)
        text_rect = label.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(label, text_rect)

    def _gui_animation(self, row, action):
        piece_color = self.RED if self.current_player == 1 else self.YELLOW
        highlight_color = self.RED_HIGHLIGHT if self.current_player == 1 else self.YELLOW_HIGHLIGHT
        
        start_y = self.square_size // 2
        target_y = int(self.top_bar_height + (row  * self.square_size) + self.square_size / 2)
        piece_x = int(action * self.square_size + self.square_size / 2)
        
        # Animate in steps - 15 pixels at a time
        for y in range(start_y, target_y, 15):
            self._handle_events()
            self._draw_static_board()
            
            # Draw falling piece
            self.pygame.draw.circle(self.screen, piece_color, (piece_x, y), self.radius)
            self.pygame.draw.circle(self.screen, highlight_color, 
                                (piece_x - 8, y - 8), self.radius // 4)
            
            self.pygame.display.flip()
            self.clock.tick(60)
        
        # Final position
        self._draw_static_board()
        self.pygame.draw.circle(self.screen, piece_color, (piece_x, target_y), self.radius)
        self.pygame.draw.circle(self.screen, highlight_color, 
                            (piece_x - 8, target_y - 8), self.radius // 4)
        self.pygame.display.flip()

    def _draw_static_board(self):
        self.screen.fill(self.BLACK)
        
        self._draw_gradient_bar()
        self._draw_board_grid()
        self._draw_column_numbers()
        self._draw_turn_indicator()

    def _render_gui(self):
        if self.pygame is None or self.screen is None:
            return
        
        self._handle_events()
        self._draw_static_board()
        if self.done:
            self._draw_game_over()
        
        self.pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.pygame is not None:
            self.pygame.quit()
            self.pygame = None


if __name__ == "__main__":
    print("Connect Four Demo")
    print("=" * 40)
    print("1. GUI Mode (render_mode='human')")
    print("2. Text Mode (render_mode='text')")
    
    choice = input("\nChoose mode (1-2): ").strip()
    
    if choice == "1":
        env = ConnectFourEnv(render_mode='human')
        print("\nPlaying in GUI mode...")
        print("Enter column numbers in terminal (0-6)")
        
        state = env.reset()
        done = False
        
        while not done:
            env.render()
            
            print(f"\nPlayer {'X (Red)' if env.current_player == 1 else 'O (Yellow)'}'s turn")
            print(f"Valid moves: {env.get_valid_actions()}")
            
            try:
                col = int(input("Enter column (0-6): "))
                state, reward, done, _, info = env.step(col)
            except ValueError:
                print("Invalid input! Enter a number.")
            except Exception as e:
                print(f"Error: {e}")
        
        env.render()
        env.close()
    
    else:
        env = ConnectFourEnv(render_mode = 'text')
        print("\nPlaying in text mode...")
        
        state = env.reset()
        done = False
        
        env.render()
        while not done:
            print(f"Player {'X' if env.current_player == 1 else 'O'}'s turn")
            print(f"Valid moves: {env.get_valid_actions()}")
            
            try:
                col = int(input("Enter column (0-6): "))
                state, reward, done, info = env.step(col)
                env.render()

            except Exception as e:
                print(f"Error: {e}")