#include "graphics/presentation/window/fsr3FrameGen.h"
#include "graphics/presentation/window/hostInput.h"
#include "graphics/presentation/window/hostInputPulse.h"

#include <SDL3/SDL.h>

#include "common/assert.h"
#include "graphics/presentation/fluidityLog.h"
#include "common/emulatorConfig.h"
#include "common/hostInputTrace.h"
#include "common/logging/log.h"
#include "libs/controller.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace Libs::Graphics {

namespace {

HostInputPulse g_input_pulse;

void ApplyHostKey(int key_code, bool down);

struct ControlInfo {
	std::string_view name;
	uint32_t         button   = 0;
	Controller::Axis axis     = Controller::Axis::AxisMax;
	bool             positive = false;
	float            touch_x  = 0.0f;

	Controller::Setting setting {};
	bool                gyro          = false;
	bool                touchpad_mode = false;
};

static constexpr std::array CONTROL_INFO = {
    ControlInfo {"L3", Controller::PAD_BUTTON_L3},
    ControlInfo {"R3", Controller::PAD_BUTTON_R3},
    ControlInfo {"Options", Controller::PAD_BUTTON_OPTIONS},
    ControlInfo {"Up", Controller::PAD_BUTTON_UP},
    ControlInfo {"Right", Controller::PAD_BUTTON_RIGHT},
    ControlInfo {"Down", Controller::PAD_BUTTON_DOWN},
    ControlInfo {"Left", Controller::PAD_BUTTON_LEFT},
    ControlInfo {"L2", Controller::PAD_BUTTON_L2},
    ControlInfo {"R2", Controller::PAD_BUTTON_R2},
    ControlInfo {"L1", Controller::PAD_BUTTON_L1},
    ControlInfo {"R1", Controller::PAD_BUTTON_R1},
    ControlInfo {"Triangle", Controller::PAD_BUTTON_TRIANGLE},
    ControlInfo {"Circle", Controller::PAD_BUTTON_CIRCLE},
    ControlInfo {"Cross", Controller::PAD_BUTTON_CROSS},
    ControlInfo {"Square", Controller::PAD_BUTTON_SQUARE},
    ControlInfo {"TouchPad", Controller::PAD_BUTTON_TOUCH_PAD, Controller::Axis::AxisMax, false,
                 0.25f},
    ControlInfo {"TouchPadRight", Controller::PAD_BUTTON_TOUCH_PAD, Controller::Axis::AxisMax,
                 false, 0.75f},
    ControlInfo {"LeftStickLeft", 0, Controller::Axis::LeftX, false},
    ControlInfo {"LeftStickRight", 0, Controller::Axis::LeftX, true},
    ControlInfo {"LeftStickUp", 0, Controller::Axis::LeftY, false},
    ControlInfo {"LeftStickDown", 0, Controller::Axis::LeftY, true},
    ControlInfo {"RightStickLeft", 0, Controller::Axis::RightX, false},
    ControlInfo {"RightStickRight", 0, Controller::Axis::RightX, true},
    ControlInfo {"RightStickUp", 0, Controller::Axis::RightY, false},
    ControlInfo {"RightStickDown", 0, Controller::Axis::RightY, true},
    ControlInfo {.name = "SpeakerVolume", .setting = Controller::Setting::SpeakerVolume},
    ControlInfo {.name = "VibrationIntensity", .setting = Controller::Setting::VibrationIntensity},
    ControlInfo {.name    = "TriggerEffectIntensity",
                 .setting = Controller::Setting::TriggerEffectIntensity},
    ControlInfo {.name = "GyroToggle", .gyro = true},
    ControlInfo {.name = "TouchPadMode", .touchpad_mode = true},
};

constexpr std::size_t INVALID_CONTROL = CONTROL_INFO.size();

struct PadSource {
	std::string_view name;
	int              button; // SDL_GamepadButton, or -1 for an analog trigger
};

constexpr std::array PAD_SOURCES = {
    PadSource {"South", SDL_GAMEPAD_BUTTON_SOUTH},
    PadSource {"East", SDL_GAMEPAD_BUTTON_EAST},
    PadSource {"West", SDL_GAMEPAD_BUTTON_WEST},
    PadSource {"North", SDL_GAMEPAD_BUTTON_NORTH},
    PadSource {"Back", SDL_GAMEPAD_BUTTON_BACK},
    PadSource {"Guide", SDL_GAMEPAD_BUTTON_GUIDE},
    PadSource {"Start", SDL_GAMEPAD_BUTTON_START},
    PadSource {"LeftStick", SDL_GAMEPAD_BUTTON_LEFT_STICK},
    PadSource {"RightStick", SDL_GAMEPAD_BUTTON_RIGHT_STICK},
    PadSource {"LeftShoulder", SDL_GAMEPAD_BUTTON_LEFT_SHOULDER},
    PadSource {"RightShoulder", SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER},
    PadSource {"DpadUp", SDL_GAMEPAD_BUTTON_DPAD_UP},
    PadSource {"DpadDown", SDL_GAMEPAD_BUTTON_DPAD_DOWN},
    PadSource {"DpadLeft", SDL_GAMEPAD_BUTTON_DPAD_LEFT},
    PadSource {"DpadRight", SDL_GAMEPAD_BUTTON_DPAD_RIGHT},
    PadSource {"Misc1", SDL_GAMEPAD_BUTTON_MISC1},
    PadSource {"RightPaddle1", SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1},
    PadSource {"LeftPaddle1", SDL_GAMEPAD_BUTTON_LEFT_PADDLE1},
    PadSource {"RightPaddle2", SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2},
    PadSource {"LeftPaddle2", SDL_GAMEPAD_BUTTON_LEFT_PADDLE2},
    PadSource {"Touchpad", SDL_GAMEPAD_BUTTON_TOUCHPAD},
    PadSource {"LeftTrigger", -1},
    PadSource {"RightTrigger", -1},
};

constexpr std::size_t LEFT_TRIGGER_SOURCE  = PAD_SOURCES.size() - 2;
constexpr std::size_t RIGHT_TRIGGER_SOURCE = PAD_SOURCES.size() - 1;
constexpr std::size_t INVALID_SOURCE       = PAD_SOURCES.size();

struct PadDefault {
	std::string_view control;
	std::string_view source;
};

constexpr std::array PAD_DEFAULTS = {
    PadDefault {"Cross", "South"},         PadDefault {"Circle", "East"},
    PadDefault {"Square", "West"},         PadDefault {"Triangle", "North"},
    PadDefault {"Options", "Start"},       PadDefault {"L3", "LeftStick"},
    PadDefault {"R3", "RightStick"},       PadDefault {"L1", "LeftShoulder"},
    PadDefault {"R1", "RightShoulder"},    PadDefault {"L2", "LeftTrigger"},
    PadDefault {"R2", "RightTrigger"},     PadDefault {"Up", "DpadUp"},
    PadDefault {"Down", "DpadDown"},       PadDefault {"Left", "DpadLeft"},
    PadDefault {"Right", "DpadRight"},     PadDefault {"TouchPad", "Touchpad"},
    PadDefault {"GyroToggle", "Back"},
};

constexpr std::string_view PAD_PREFIX = "Pad:";

struct Binding {
	SDL_Keycode key          = SDLK_UNKNOWN;
	uint8_t     mouse_button = 0;
	std::size_t control      = INVALID_CONTROL;
};

constexpr int              MOUSE_POLL_INTERVAL_MS = 33;
constexpr uint64_t         CURSOR_IDLE_HIDE_MS    = 2000;
constexpr std::string_view MOUSE_SENSITIVITY      = "MouseSensitivity=";
constexpr std::string_view TOUCH_PARTICLE_SKIN    = "TouchParticleSkin=";

struct MouseJoystickState {
	bool     enabled   = false;
	bool     output    = false;
	uint64_t next_poll = 0;
};

MouseJoystickState g_mouse;
SDL_Window*        g_mouse_window   = nullptr;
uint64_t           g_cursor_hide_at = 0;

struct MouseTouchPadState {
	bool     enabled        = false;
	bool     cursor_visible = true;
	bool     drawing        = false;
	bool     position_valid = false;
	float    x              = 0.5f;
	float    y              = 0.5f;
	uint64_t revision       = 0;
	uint64_t gesture        = 0;
};

struct GamepadTouchPadState {
	bool     enabled   = false;
	bool     drawing   = false;
	int      id        = -1;
	int      x_axis    = 128;
	int      y_axis    = 128;
	float    x         = 0.5f;
	float    y         = 0.5f;
	uint64_t next_poll = 0;
	uint64_t last_poll = 0;
};

constexpr int   GAMEPAD_TOUCHPAD_POLL_MS = 8;
constexpr int   GAMEPAD_TOUCHPAD_DEADZONE = 18;
constexpr float GAMEPAD_TOUCHPAD_SPEED    = 0.8f;

MouseTouchPadState   g_mouse_touchpad;
GamepadTouchPadState g_gamepad_touchpad;
std::mutex           g_touchpad_mutex;
std::unordered_map<int, std::array<bool, 2>> g_pad_trigger_pressed;

std::size_t ControlFromName(std::string_view name) {
	const auto info = std::find_if(CONTROL_INFO.begin(), CONTROL_INFO.end(),
	                               [name](const auto& item) { return item.name == name; });
	return static_cast<std::size_t>(std::distance(CONTROL_INFO.begin(), info));
}

SDL_Keycode NormalizeKey(SDL_Keycode key) {
	switch (key) {
		case SDLK_RSHIFT: return SDLK_LSHIFT;
		case SDLK_RCTRL: return SDLK_LCTRL;
		case SDLK_RALT: return SDLK_LALT;
		case SDLK_RGUI: return SDLK_LGUI;
		default: return key;
	}
}

uint8_t MouseButtonFromName(std::string_view name) {
	static constexpr std::array buttons = {
	    std::pair {std::string_view("Mouse:Left"), uint8_t {SDL_BUTTON_LEFT}},
	    std::pair {std::string_view("Mouse:Middle"), uint8_t {SDL_BUTTON_MIDDLE}},
	    std::pair {std::string_view("Mouse:Right"), uint8_t {SDL_BUTTON_RIGHT}},
	    std::pair {std::string_view("Mouse:X1"), uint8_t {SDL_BUTTON_X1}},
	    std::pair {std::string_view("Mouse:X2"), uint8_t {SDL_BUTTON_X2}},
	};

	const auto button = std::find_if(buttons.begin(), buttons.end(),
	                                 [name](const auto& item) { return item.first == name; });
	return button != buttons.end() ? button->second : 0;
}

bool Conflicts(const Binding& first, const Binding& second) {
	return first.control == second.control ||
	       (first.key != SDLK_UNKNOWN && first.key == second.key) ||
	       (first.mouse_button != 0 && first.mouse_button == second.mouse_button);
}

class InputMap {
public:
	InputMap() {
		for (const auto& value: Config::GetKeymap()) {
			const std::string_view entry = value;
			if (entry == "Keyboard=Disabled") {
				m_custom = true;
				continue;
			}
			if (entry == "Pad=Disabled") {
				m_custom_pad = true;
				continue;
			}
			if (entry.starts_with(MOUSE_SENSITIVITY)) {
				const float sensitivity =
				    std::strtof(value.c_str() + MOUSE_SENSITIVITY.size(), nullptr);
				m_mouse_sensitivity =
				    std::clamp(std::isfinite(sensitivity) ? sensitivity : 1.0f, 0.1f, 5.0f);
				continue;
			}
			if (entry.starts_with(TOUCH_PARTICLE_SKIN) || entry.starts_with("TouchCursorImage=")) {
				continue;
			}
			if (entry.starts_with(PAD_PREFIX)) {
				AddPadEntry(entry.substr(PAD_PREFIX.size()), value);
				continue;
			}
			const auto split = entry.find('=');

			Binding binding;
			if (split != std::string_view::npos) {
				binding.control       = ControlFromName(entry.substr(0, split));
				const auto host_input = entry.substr(split + 1);
				binding.mouse_button  = MouseButtonFromName(host_input);
				if (binding.mouse_button == 0) {
					binding.key = NormalizeKey(SDL_GetKeyFromName(std::string(host_input).c_str()));
				}
			}

			const bool reserved = binding.key == SDLK_ESCAPE || binding.key == SDLK_F1 ||
			                      binding.key == SDLK_F2 || binding.key == SDLK_F7 ||
			                      binding.key == SDLK_F8 || binding.key == SDLK_F11;
			if (binding.control == INVALID_CONTROL || reserved ||
			    (binding.key == SDLK_UNKNOWN && binding.mouse_button == 0)) {
				EXIT("Invalid input mapping: %s\n", value.c_str());
			}
			Add(binding);
		}
		if (!m_custom_pad) {
			for (const auto& item: PAD_DEFAULTS) {
				AssignPad(ControlFromName(item.control), SourceFromName(item.source));
			}
		}
	}
	[[nodiscard]] bool  Custom() const { return m_custom; }
	[[nodiscard]] float MouseSensitivity() const { return m_mouse_sensitivity; }

	[[nodiscard]] std::size_t PadControl(std::size_t source) const {
		return source < m_pad_controls.size() ? m_pad_controls[source] : INVALID_CONTROL;
	}

	static std::size_t SourceFromButton(int button) {
		const auto item = std::find_if(PAD_SOURCES.begin(), PAD_SOURCES.end(),
		                               [button](const auto& s) { return s.button == button; });
		return static_cast<std::size_t>(std::distance(PAD_SOURCES.begin(), item));
	}

	[[nodiscard]] std::size_t FindKey(int key_code) const {
		key_code = NormalizeKey(static_cast<SDL_Keycode>(key_code));
		const auto binding =
		    std::find_if(m_bindings.begin(), m_bindings.begin() + m_size,
		                 [key_code](const auto& item) { return item.key == key_code; });
		return binding != m_bindings.begin() + m_size ? binding->control : INVALID_CONTROL;
	}

	[[nodiscard]] std::size_t FindMouseButton(uint8_t mouse_button) const {
		const auto binding = std::find_if(
		    m_bindings.begin(), m_bindings.begin() + m_size,
		    [mouse_button](const auto& item) { return item.mouse_button == mouse_button; });
		return binding != m_bindings.begin() + m_size ? binding->control : INVALID_CONTROL;
	}

private:
	static std::size_t SourceFromName(std::string_view name) {
		const auto item = std::find_if(PAD_SOURCES.begin(), PAD_SOURCES.end(),
		                               [name](const auto& s) { return s.name == name; });
		return static_cast<std::size_t>(std::distance(PAD_SOURCES.begin(), item));
	}

	void AssignPad(std::size_t control, std::size_t source) {
		for (auto& existing: m_pad_controls) {
			if (existing == control) {
				existing = INVALID_CONTROL;
			}
		}
		m_pad_controls[source] = control;
	}

	void AddPadEntry(std::string_view entry, const std::string& original) {
		const auto split   = entry.find('=');
		const auto control = split != std::string_view::npos ? ControlFromName(entry.substr(0, split))
		                                                     : INVALID_CONTROL;
		const auto source  = split != std::string_view::npos ? SourceFromName(entry.substr(split + 1))
		                                                     : INVALID_SOURCE;
		if (control == INVALID_CONTROL || source == INVALID_SOURCE) {
			EXIT("Invalid input mapping: %s\n", original.c_str());
		}
		const auto& info = CONTROL_INFO[control];
		if (info.button == 0 && !info.gyro && info.axis != Controller::Axis::AxisMax) {
			EXIT("Invalid input mapping: %s\n", original.c_str());
		}
		m_custom_pad = true;
		AssignPad(control, source);
	}
	void Add(const Binding& binding) {
		m_custom = true;
		for (std::size_t index = 0; index < m_size;) {
			if (Conflicts(m_bindings[index], binding)) {
				m_bindings[index] = m_bindings[--m_size];
			} else {
				index++;
			}
		}
		EXIT_IF(m_size >= m_bindings.size());
		m_bindings[m_size++] = binding;
	}

	std::array<Binding, CONTROL_INFO.size()> m_bindings {};
	std::array<std::size_t, PAD_SOURCES.size()> m_pad_controls = [] {
		std::array<std::size_t, PAD_SOURCES.size()> a {};
		a.fill(INVALID_CONTROL);
		return a;
	}();
	bool                                     m_custom = false;
	bool                                     m_custom_pad = false;
	std::size_t                              m_size              = 0;
	float                                    m_mouse_sensitivity = 1.0f;
};

const InputMap& GetInputMap() {
	static const InputMap map;
	return map;
}

void SetButton(uint32_t button, bool down) {
	if (button == Controller::PAD_BUTTON_L2) {
		Controller::SetAxis(Controller::HOST_INPUT_CONTROLLER_ID, Controller::Axis::TriggerLeft,
		                    down ? 255 : 0);
	} else if (button == Controller::PAD_BUTTON_R2) {
		Controller::SetAxis(Controller::HOST_INPUT_CONTROLLER_ID, Controller::Axis::TriggerRight,
		                    down ? 255 : 0);
	} else if (button != 0) {
		Controller::SetButton(Controller::HOST_INPUT_CONTROLLER_ID, button, down);
	}
}

void SetTouchPad(float x, bool down) {
	Controller::SetTouchPad(Controller::HOST_INPUT_CONTROLLER_ID, 0, down, x, 0.5f);
}

void SetSwipeTouchPad(bool down, float x, float y) {
	Controller::SetTouchPad(Controller::HOST_INPUT_CONTROLLER_ID, 1, down, x, y);
}

void RecordMouseTouchPadPosition(float x, float y) {
	int width  = 0;
	int height = 0;
	if (g_mouse_window != nullptr) {
		SDL_GetWindowSize(g_mouse_window, &width, &height);
	}
	if (width > 0 && height > 0) {
		x = std::clamp(x / static_cast<float>(width), 0.0f, 1.0f);
		y = std::clamp(y / static_cast<float>(height), 0.0f, 1.0f);
	} else {
		x = 0.5f;
		y = 0.5f;
	}

	std::scoped_lock lock(g_touchpad_mutex);
	if (!g_mouse_touchpad.enabled) {
		return;
	}
	g_mouse_touchpad.x              = x;
	g_mouse_touchpad.y              = y;
	g_mouse_touchpad.position_valid = true;
	g_mouse_touchpad.revision++;
}

bool TouchPadModeActive() {
	std::scoped_lock lock(g_touchpad_mutex);
	return g_mouse_touchpad.enabled || g_gamepad_touchpad.enabled;
}

uint32_t DefaultKeyboardButton(int key_code) {
	switch (NormalizeKey(static_cast<SDL_Keycode>(key_code))) {
		case SDLK_UP: return Controller::PAD_BUTTON_UP;
		case SDLK_LEFT: return Controller::PAD_BUTTON_LEFT;
		case SDLK_DOWN: return Controller::PAD_BUTTON_DOWN;
		case SDLK_RIGHT: return Controller::PAD_BUTTON_RIGHT;
		case SDLK_J: return Controller::PAD_BUTTON_CROSS;
		case SDLK_I: return Controller::PAD_BUTTON_TRIANGLE;
		case SDLK_K: return Controller::PAD_BUTTON_SQUARE;
		case SDLK_L: return Controller::PAD_BUTTON_CIRCLE;
		case SDLK_Q: return Controller::PAD_BUTTON_L1;
		case SDLK_E: return Controller::PAD_BUTTON_R1;
		case SDLK_Z: return Controller::PAD_BUTTON_L2;
		case SDLK_C: return Controller::PAD_BUTTON_R2;
		case SDLK_LSHIFT: return Controller::PAD_BUTTON_L3;
		case SDLK_LCTRL: return Controller::PAD_BUTTON_R3;
		case SDLK_RETURN:
		case SDLK_RETURN2: return Controller::PAD_BUTTON_OPTIONS;
		default: return 0;
	}
}

struct StickKeys {
	bool left  = false;
	bool right = false;
	bool up    = false;
	bool down  = false;
};

void SetStickAxis(Controller::Axis axis, bool negative, bool positive) {
	int value = 128;
	if (negative && !positive) {
		value = 0;
	} else if (positive && !negative) {
		value = 255;
	}
	Controller::SetAxis(Controller::HOST_INPUT_CONTROLLER_ID, axis, value);
}

void CycleSettingOnPress(Controller::Setting setting, bool down) {
	if (down) {
		Controller::CycleSetting(setting);
	}
}

void SetControl(std::size_t control, bool down) {
	if (control == INVALID_CONTROL) {
		return;
	}

	const auto& info = CONTROL_INFO[control];
	if (info.button == Controller::PAD_BUTTON_TOUCH_PAD) {
		SetTouchPad(info.touch_x, down);
		return;
	}
	if (info.button != 0) {
		SetButton(info.button, down);
		return;
	}
	if (info.gyro) {
		if (down) {
			Controller::ToggleGyroEmulation(Controller::HOST_INPUT_CONTROLLER_ID);
		}
		return;
	}
	if (info.touchpad_mode) {
		if (down) {
			HostInputToggleTouchPadMode();
		}
		return;
	}
	if (info.axis == Controller::Axis::AxisMax) {
		CycleSettingOnPress(info.setting, down);
		return;
	}

	struct AxisKeys {
		bool negative = false;
		bool positive = false;
	};
	static std::array<AxisKeys, 4> axes;

	const auto axis = static_cast<std::size_t>(info.axis);
	EXIT_IF(axis >= axes.size());
	if (info.positive) {
		axes[axis].positive = down;
	} else {
		axes[axis].negative = down;
	}
	SetStickAxis(info.axis, axes[axis].negative, axes[axis].positive);
}

void DefaultKeyboardInput(int key_code, bool down) {
	static StickKeys left;
	static StickKeys right;

	switch (NormalizeKey(static_cast<SDL_Keycode>(key_code))) {
		case SDLK_BACKSPACE: SetTouchPad(0.25f, down); return;
		case SDLK_F6:
			if (down) {
				HostInputToggleTouchPadMode();
			}
			return;
		case SDLK_TAB: SetTouchPad(0.75f, down); return;
		case SDLK_1: CycleSettingOnPress(Controller::Setting::SpeakerVolume, down); return;
		case SDLK_2: CycleSettingOnPress(Controller::Setting::VibrationIntensity, down); return;
		case SDLK_3: CycleSettingOnPress(Controller::Setting::TriggerEffectIntensity, down); return;
		case SDLK_A:
			left.left = down;
			SetStickAxis(Controller::Axis::LeftX, left.left, left.right);
			return;
		case SDLK_D:
			left.right = down;
			SetStickAxis(Controller::Axis::LeftX, left.left, left.right);
			return;
		case SDLK_W:
			left.up = down;
			SetStickAxis(Controller::Axis::LeftY, left.up, left.down);
			return;
		case SDLK_S:
			left.down = down;
			SetStickAxis(Controller::Axis::LeftY, left.up, left.down);
			return;
		case SDLK_F:
			right.left = down;
			SetStickAxis(Controller::Axis::RightX, right.left, right.right);
			return;
		case SDLK_H:
			right.right = down;
			SetStickAxis(Controller::Axis::RightX, right.left, right.right);
			return;
		case SDLK_T:
			right.up = down;
			SetStickAxis(Controller::Axis::RightY, right.up, right.down);
			return;
		case SDLK_G:
			right.down = down;
			SetStickAxis(Controller::Axis::RightY, right.up, right.down);
			return;
		default: SetButton(DefaultKeyboardButton(key_code), down); return;
	}
}

void ApplyHostKey(int key_code, bool down) {
	Common::HostInputTrace("apply-key", key_code, down);
	const auto& map = GetInputMap();
	if (map.Custom()) {
		SetControl(map.FindKey(key_code), down);
	} else {
		DefaultKeyboardInput(key_code, down);
	}
}

void MouseToJoystick(float delta_x, float delta_y) {
	const double distance = std::hypot(delta_x, delta_y);
	const double scale =
	    std::clamp(distance * GetInputMap().MouseSensitivity() + 16.0, 64.0, 128.0) / distance;
	const auto map_axis = [scale](float delta) {
		return std::clamp(128 + static_cast<int>(std::lround(delta * scale)), 0, 255);
	};
	Controller::SetRightStick(Controller::HOST_INPUT_CONTROLLER_ID, map_axis(delta_x),
	                          map_axis(delta_y));
}

void CenterMouseStick() {
	if (!g_mouse.output) {
		return;
	}
	Controller::SetRightStick(Controller::HOST_INPUT_CONTROLLER_ID, 128, 128);
	g_mouse.output = false;
}

bool SetRelativeMouseMode(bool enabled) {
	if (SDL_SetWindowRelativeMouseMode(g_mouse_window, enabled)) {
		return true;
	}
	LOGF("Mouse-to-joystick relative mode failed: %s\n", SDL_GetError());
	return false;
}

int PollMouse(uint64_t now_ms) {
	if (now_ms < g_mouse.next_poll) {
		return static_cast<int>(g_mouse.next_poll - now_ms);
	}
	g_mouse.next_poll = now_ms + MOUSE_POLL_INTERVAL_MS;

	float delta_x = 0;
	float delta_y = 0;
	SDL_GetRelativeMouseState(&delta_x, &delta_y);
	if (delta_x == 0 && delta_y == 0) {
		CenterMouseStick();
	} else {
		MouseToJoystick(delta_x, delta_y);
		g_mouse.output = true;
	}
	return MOUSE_POLL_INTERVAL_MS;
}

bool IsCursorActivity(const SDL_Event& event) {
	SDL_MouseID which;
	switch (event.type) {
		case SDL_EVENT_MOUSE_MOTION:
			if (event.motion.xrel == 0.0f && event.motion.yrel == 0.0f) {
				return false;
			}
			which = event.motion.which;
			break;
		case SDL_EVENT_MOUSE_BUTTON_DOWN:
		case SDL_EVENT_MOUSE_BUTTON_UP: which = event.button.which; break;
		case SDL_EVENT_MOUSE_WHEEL: which = event.wheel.which; break;
		default: return false;
	}
	return which != SDL_TOUCH_MOUSEID && which != SDL_PEN_MOUSEID;
}

std::array<int, 2> GamepadLeftStick(int id) {
	auto* pad = SDL_GetGamepadFromID(id);
	if (pad == nullptr) {
		return {128, 128};
	}
	const auto value = [pad](SDL_GamepadAxis axis) {
		return Controller::controller_get_axis(SDL_JOYSTICK_AXIS_MIN, SDL_JOYSTICK_AXIS_MAX,
		                                       SDL_GetGamepadAxis(pad, axis));
	};
	return {value(SDL_GAMEPAD_AXIS_LEFTX), value(SDL_GAMEPAD_AXIS_LEFTY)};
}

void ApplyPadControl(int id, std::size_t control, std::size_t source, bool down) {
	if (control == INVALID_CONTROL) {
		return;
	}
	const auto& info = CONTROL_INFO[control];
	if (info.gyro) {
		if (down) {
			Controller::ToggleGyroEmulation(id);
		}
	} else if (info.touchpad_mode) {
		if (down) {
			HostInputToggleGamepadTouchPad(id);
		}
	} else if (info.button == Controller::PAD_BUTTON_CROSS && HostInputGamepadDrawButton(id, down)) {
		return;
	} else if (info.button == Controller::PAD_BUTTON_TOUCH_PAD) {
		// Real touchpads report their own finger position; other inputs emulate a click.
		if (source != PAD_SOURCES.size() && PAD_SOURCES[source].button != SDL_GAMEPAD_BUTTON_TOUCHPAD) {
			Controller::SetTouchPad(id, 0, down, info.touch_x != 0.0f ? info.touch_x : 0.5f, 0.5f);
		}
		Controller::SetButton(id, Controller::PAD_BUTTON_TOUCH_PAD, down);
	} else if (info.button == Controller::PAD_BUTTON_L2) {
		Controller::SetAxis(id, Controller::Axis::TriggerLeft, down ? 255 : 0);
	} else if (info.button == Controller::PAD_BUTTON_R2) {
		Controller::SetAxis(id, Controller::Axis::TriggerRight, down ? 255 : 0);
	} else if (info.button != 0) {
		Controller::SetButton(id, info.button, down);
	} else if (info.axis == Controller::Axis::AxisMax) {
		CycleSettingOnPress(info.setting, down);
	}
}

int PollGamepadTouchPad(uint64_t now_ms) {
	float x       = 0;
	float y       = 0;
	int   id      = -1;
	bool  drawing = false;
	{
		std::scoped_lock lock(g_touchpad_mutex);
		if (!g_gamepad_touchpad.enabled) {
			return -1;
		}
		if (now_ms < g_gamepad_touchpad.next_poll) {
			return static_cast<int>(g_gamepad_touchpad.next_poll - now_ms);
		}

		const float dt = g_gamepad_touchpad.last_poll == 0 || now_ms <= g_gamepad_touchpad.last_poll
		                     ? 0.008f
		                     : std::min(static_cast<float>(now_ms - g_gamepad_touchpad.last_poll) *
		                                    0.001f,
		                                0.05f);
		g_gamepad_touchpad.last_poll = now_ms;
		g_gamepad_touchpad.next_poll = now_ms + GAMEPAD_TOUCHPAD_POLL_MS;
		const auto velocity          = [](int axis) {
            const int offset = axis - 128;
            if (std::abs(offset) <= GAMEPAD_TOUCHPAD_DEADZONE) {
                return 0.0f;
            }
            const int adjusted =
                offset > 0 ? offset - GAMEPAD_TOUCHPAD_DEADZONE : offset + GAMEPAD_TOUCHPAD_DEADZONE;
            return static_cast<float>(adjusted) / static_cast<float>(127 - GAMEPAD_TOUCHPAD_DEADZONE);
		};
		const float old_x = g_gamepad_touchpad.x;
		const float old_y = g_gamepad_touchpad.y;
		g_gamepad_touchpad.x = std::clamp(
		    old_x + velocity(g_gamepad_touchpad.x_axis) * GAMEPAD_TOUCHPAD_SPEED * dt, 0.0f, 1.0f);
		g_gamepad_touchpad.y = std::clamp(
		    old_y + velocity(g_gamepad_touchpad.y_axis) * GAMEPAD_TOUCHPAD_SPEED * dt, 0.0f, 1.0f);
		if (old_x != g_gamepad_touchpad.x || old_y != g_gamepad_touchpad.y) {
			g_mouse_touchpad.revision++;
		}
		id      = g_gamepad_touchpad.id;
		x       = g_gamepad_touchpad.x;
		y       = g_gamepad_touchpad.y;
		drawing = g_gamepad_touchpad.drawing;
	}
	if (drawing) {
		Controller::SetTouchPad(id, 1, true, x, y);
	}
	return GAMEPAD_TOUCHPAD_POLL_MS;
}

} // namespace

void HostInputPadButton(int id, int button, bool down) {
	if (button == SDL_GAMEPAD_BUTTON_LEFT_STICK || button == SDL_GAMEPAD_BUTTON_RIGHT_STICK) {
		FluidityNotePadButton(id, button == SDL_GAMEPAD_BUTTON_LEFT_STICK, down);
	}
	const auto  source = InputMap::SourceFromButton(button);
	const auto& map    = GetInputMap();
	ApplyPadControl(id, map.PadControl(source), source, down);
}

void HostInputPadTrigger(int id, bool left, int value) {
	const auto  source     = left ? LEFT_TRIGGER_SOURCE : RIGHT_TRIGGER_SOURCE;
	const auto  control    = GetInputMap().PadControl(source);
	if (control == INVALID_CONTROL) {
		return;
	}
	const auto& info = CONTROL_INFO[control];
	if (info.button == Controller::PAD_BUTTON_L2) {
		Controller::SetAxis(id, Controller::Axis::TriggerLeft, value);
	} else if (info.button == Controller::PAD_BUTTON_R2) {
		Controller::SetAxis(id, Controller::Axis::TriggerRight, value);
	} else {
		auto& state = g_pad_trigger_pressed[id][left ? 0 : 1];
		const bool down = state ? value > 32 : value > 96;
		if (down != state) {
			state = down;
			ApplyPadControl(id, control, source, down);
		}
	}
}

void HostInputDisconnectGamepad(int id) {
	g_pad_trigger_pressed.erase(id);
	HostInputDisconnectGamepadTouchPad(id);
}

namespace {

// Astro Bot PC (KYTY_PAD_FILE=<fichier>, off by default): internal pad driver for unattended tests. A thread
// reads commands from the file and sets the host controller directly (Controller::SetAxis/SetButton), so the
// character moves without window focus, SendInput or an unlocked Windows session. Commands, separated by ';'
// or new lines: "maintenir <K>[+<K>...] <ms>" (hold), "touche <K>[+<K>...] [ms=120]" (press), "attendre <ms>",
// "boucle" (what follows repeats until the file changes). Keys: W A S D (left stick), F H T G (right stick),
// J=Cross K=Square I=Triangle L=Circle Q=L1 E=R1 Z=L2 C=R2 ENTREE=Options. The file is re-read when it
// changes (a new script replaces the current one at once); an empty or missing file releases everything.
// v2 (18/11): "aller <x> <z> [tolerance=0.05] [max_ms=20000]" steers the left stick toward the camera
// position (x, z), read from KYTY_FLUIDITE_CAMERA (FluidityCamera), until the camera is within
// `tolerance`: a return to a recorded place that does not depend on the timing of a replay. The camera
// trails the character, so the target is a camera position as recorded in trace.csv / journal.csv.
// "tenir <ms>" then stays on that target. A script without "boucle" now runs once (v1 restarted it).
// v3 (19/11), the replay of a recorded path (tools: preparer-retour.py, from trace.csv and manette.csv):
//   "passer <x> <z> [tolerance=0.6] [max_ms=8000]": like "aller", but at full speed and without stopping
//   (a waypoint of the path); "bouton <K>[+<K>...] <ms>": presses buttons for <ms> while the next steps
//   run (a jump during the walk); "attendre-pres <x> <z> [tolerance=1] [max_ms=300000]": waits, stick
//   released, until the camera is within tolerance of (x, z) (the level is loaded, at its start);
//   "attendre-calme [ms=2000] [max_ms=300000]": waits until the camera is valid and has moved less than 0.05 for
//   <ms> (a level started directly opens on a fly-over: the game starts when the camera settles on Astro);
//   "rejoindre": skips to the recorded waypoint ("passer") nearest the camera: the first within 3 units, else
//   the nearest one (the recorded path may start before the place the level puts Astro at).
class PadScript {
public:
	void Start() {
		const char* path = std::getenv("KYTY_PAD_FILE");
		if (path == nullptr || *path == '\0' || m_thread.joinable()) {
			return;
		}
		m_path = path;
		m_stop = false;
		m_thread = std::thread([this] { Run(); });
		std::printf("Kyty pilote interne : commandes lues dans %s (KYTY_PAD_FILE)\n", m_path.c_str());
		std::fflush(stdout);
	}
	void Stop() {
		if (m_thread.joinable()) {
			m_stop = true;
			m_thread.join();
		}
		ReleaseAll();
	}

private:
	struct Step {
		int      kind = 0; // 0 wait, 1 hold, 2 go to a camera position, 3 hold the target, 4 pass, 5 button,
		                   // 6 wait near a camera position, 7 wait for a settled camera, 8 join the path
		uint32_t keys = 0; // bit per key index
		uint32_t ms   = 0;
		float    x = 0, z = 0, tolerance = 0.05f;
		uint32_t max_ms = 0; // "attendre-calme"
		std::string file;    // "rejouer" (kind 9): commandes.csv, frames from..to
		uint64_t    from = 0, to = 0;
	};

	static void ApplyStick(float side, float forward) {
		const int  id    = Controller::HOST_INPUT_CONTROLLER_ID;
		const auto clamp = [](float v) { return std::clamp(static_cast<int>(128.0f + v * 127.0f), 0, 255); };
		Controller::SetAxis(id, Controller::Axis::LeftX, clamp(side));
		Controller::SetAxis(id, Controller::Axis::LeftY, clamp(-forward)); // up = 0
	}

	// "aller": false if the file changed or the driver stops (true also on timeout, logged).
	bool Go(const Step& s) {
		const bool pass = s.kind == 4;
		m_target     = s;
		m_has_target = true;
		float pos[3] {}, dir[3] {};
		if (!Libs::Graphics::FluidityCamera(pos, dir)) {
			std::printf("Kyty pilote interne : aller %.3f %.3f impossible (camera non lue, KYTY_FLUIDITE_CAMERA)\n", s.x, s.z);
			std::fflush(stdout);
			return true;
		}
		m_active = true;
		if (m_side == 0) {
			// The stick's right is -(-fz, fx) in the camera's frame for Astro Bot (18/11: arrived at 0.051 30.214 for
			// 0.05 30 with -1; a push-right calibration gave -1 then +1 and was dropped). KYTY_PAD_SIDE=1 flips it.
			const char* side = std::getenv("KYTY_PAD_SIDE");
			m_side           = side != nullptr && std::atoi(side) > 0 ? 1 : -1;
		}
		const auto start = std::chrono::steady_clock::now();
		for (;;) {
			UpdateButtons();
			if (!Libs::Graphics::FluidityCamera(pos, dir)) {
				ApplyStick(0, 0);
				return Sleep(50);
			}
			const float dx = s.x - pos[0], dz = s.z - pos[2];
			const float dist = std::sqrt(dx * dx + dz * dz);
			if (dist <= s.tolerance) {
				break;
			}
			if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(s.ms)) {
				std::printf("Kyty pilote interne : %s %.3f %.3f : delai depasse a %.3f\n", pass ? "passer" : "aller", s.x,
				            s.z, dist);
				std::fflush(stdout);
				m_timeouts++;
				break;
			}
			float fx = dir[0], fz = dir[2];
			const float fl = std::sqrt(fx * fx + fz * fz);
			if (fl < 1e-4f) {
				fx = 0;
				fz = 1;
			} else {
				fx /= fl;
				fz /= fl;
			}
			const float nx = dx / dist, nz = dz / dist;
			const float forward = nx * fx + nz * fz;
			const float side    = static_cast<float>(m_side) * (-nx * fz + nz * fx);
			// Slow down over the last 3 units (18/11: at full speed Astro ran past 0.05 30 and slid down to z 11).
			const float speed = pass ? 1.0f : std::clamp(dist / 3.0f, 0.3f, 1.0f);
			ApplyStick(side * speed, forward * speed);
			if (!Sleep(50)) {
				return false;
			}
		}
		if (pass) {
			m_passed++;
			return true; // keep the stick: the next waypoint steers on
		}
		ApplyStick(0, 0);
		float p2[3] {}, d2[3] {};
		if (!m_quiet && Libs::Graphics::FluidityCamera(p2, d2)) {
			std::printf("Kyty pilote interne : arrive a %.3f %.3f %.3f (cible %.3f %.3f)\n", p2[0], p2[1], p2[2], s.x, s.z);
			std::fflush(stdout);
		}
		return true;
	}

	// "bouton": the presses still running, and the button keys they hold.
	struct Press {
		uint32_t                              keys = 0;
		std::chrono::steady_clock::time_point end {};
	};
	std::vector<Press> m_presses;
	uint32_t           m_button_keys = 0; // what UpdateButtons last applied (key bits 8+)
	uint32_t           m_passed      = 0; // "passer" waypoints reached
	uint32_t           m_timeouts    = 0; // "aller"/"passer" timeouts

	static constexpr uint32_t BUTTON_KEYS = ~0xffu; // keys 8+ are buttons, 0-7 sticks

	static void ApplyButtons(uint32_t keys) {
		static constexpr uint32_t BUTTONS[] = {Controller::PAD_BUTTON_CROSS,    Controller::PAD_BUTTON_SQUARE,
		                                       Controller::PAD_BUTTON_TRIANGLE, Controller::PAD_BUTTON_CIRCLE,
		                                       Controller::PAD_BUTTON_L1,       Controller::PAD_BUTTON_R1,
		                                       Controller::PAD_BUTTON_L2,       Controller::PAD_BUTTON_R2,
		                                       Controller::PAD_BUTTON_OPTIONS};
		for (int b = 0; b < static_cast<int>(std::size(BUTTONS)); b++) {
			Controller::SetButton(Controller::HOST_INPUT_CONTROLLER_ID, BUTTONS[b], ((keys >> (8 + b)) & 1u) != 0);
		}
	}

	// Applies the buttons of the running presses (only on a change).
	void UpdateButtons() {
		const auto now  = std::chrono::steady_clock::now();
		uint32_t   keys = 0;
		std::erase_if(m_presses, [&](const Press& p) { return p.end <= now; });
		for (const auto& p: m_presses) {
			keys |= p.keys;
		}
		keys &= BUTTON_KEYS;
		if (keys != m_button_keys) {
			ApplyButtons(keys);
			m_button_keys = keys;
			m_active      = true;
		}
	}

	// "attendre-calme": false if the file changed or the driver stops.
	bool WaitCalm(const Step& s) {
		ApplyStick(0, 0);
		const auto start = std::chrono::steady_clock::now();
		float      last[3] {};
		bool       have = false;
		auto       still_since = start;
		for (;;) {
			float pos[3] {}, dir[3] {};
			const auto now = std::chrono::steady_clock::now();
			const bool ok  = Libs::Graphics::FluidityCamera(pos, dir) &&
			                !(std::fabs(pos[0]) < 1e-3f && std::fabs(pos[1]) < 1e-3f && std::fabs(pos[2]) < 1e-3f);
			if (!ok) {
				have = false;
			} else if (!have || std::fabs(pos[0] - last[0]) + std::fabs(pos[1] - last[1]) + std::fabs(pos[2] - last[2]) > 0.05f) {
				std::copy(pos, pos + 3, last);
				have        = true;
				still_since = now;
			} else if (now - still_since >= std::chrono::milliseconds(s.ms)) {
				std::printf("Kyty pilote interne : attendre-calme : camera posee a %.3f %.3f %.3f apres %lld s\n", pos[0],
				            pos[1], pos[2],
				            static_cast<long long>(std::chrono::duration_cast<std::chrono::seconds>(now - start).count()));
				std::fflush(stdout);
				return true;
			}
			if (now - start > std::chrono::milliseconds(s.max_ms)) {
				std::printf("Kyty pilote interne : attendre-calme : delai depasse\n");
				std::fflush(stdout);
				return true;
			}
			if (!Sleep(100)) {
				return false;
			}
		}
	}

	// "rejoindre": the index of the waypoint to continue from (i: the step after "rejoindre").
	size_t Join(size_t i) const {
		float pos[3] {}, dir[3] {};
		if (!Libs::Graphics::FluidityCamera(pos, dir)) {
			return i;
		}
		size_t best = i, first = SIZE_MAX;
		float  best_d = 1e30f;
		for (size_t k = i; k < m_steps.size(); k++) {
			if (m_steps[k].kind != 4 && m_steps[k].kind != 5) {
				break;
			}
			if (m_steps[k].kind != 4) {
				continue;
			}
			const float d = std::hypot(m_steps[k].x - pos[0], m_steps[k].z - pos[2]);
			if (d < best_d) {
				best_d = d;
				best   = k;
			}
			if (first == SIZE_MAX && d <= 3.0f) {
				first = k;
			}
		}
		const size_t to = first != SIZE_MAX ? first : best;
		std::printf("Kyty pilote interne : rejoindre : point %zu sur %zu, a %.2f de la camera %.3f %.3f\n", to - i,
		            m_steps.size() - i, std::hypot(m_steps[to].x - pos[0], m_steps[to].z - pos[2]), pos[0], pos[2]);
		std::fflush(stdout);
		return to;
	}

	// "attendre-pres": false if the file changed or the driver stops.
	bool WaitNear(const Step& s) {
		ApplyStick(0, 0);
		const auto start = std::chrono::steady_clock::now();
		for (;;) {
			float pos[3] {}, dir[3] {};
			if (Libs::Graphics::FluidityCamera(pos, dir)) {
				const float dx = s.x - pos[0], dz = s.z - pos[2];
				const float dist = std::sqrt(dx * dx + dz * dz);
				if (dist <= s.tolerance) {
					std::printf("Kyty pilote interne : attendre-pres %.3f %.3f : atteint apres %lld s (camera %.3f %.3f)\n",
					            s.x, s.z,
					            static_cast<long long>(std::chrono::duration_cast<std::chrono::seconds>(
					                                       std::chrono::steady_clock::now() - start)
					                                       .count()),
					            pos[0], pos[2]);
					std::fflush(stdout);
					return true;
				}
			}
			if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(s.ms)) {
				std::printf("Kyty pilote interne : attendre-pres %.3f %.3f : delai depasse\n", s.x, s.z);
				std::fflush(stdout);
				return true;
			}
			if (!Sleep(200)) {
				return false;
			}
		}
	}

	// "tenir <ms>": stays on the last "aller" target, steering back when the camera drifts past twice the
	// tolerance (slopes, pushes); without a target, a plain wait.
	bool Hold(uint32_t ms) {
		if (!m_has_target) {
			return Sleep(ms);
		}
		const auto end   = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
		uint32_t   backs = 0;
		while (std::chrono::steady_clock::now() < end) {
			float pos[3] {}, dir[3] {};
			if (Libs::Graphics::FluidityCamera(pos, dir)) {
				const float dx = m_target.x - pos[0], dz = m_target.z - pos[2];
				if (std::sqrt(dx * dx + dz * dz) > 2.0f * m_target.tolerance) {
					Step back = m_target;
					back.ms   = 5000;
					m_quiet   = true;
					const bool ok = Go(back);
					m_quiet       = false;
					backs++;
					if (!ok) {
						return false;
					}
					continue;
				}
			}
			if (!Sleep(100)) {
				return false;
			}
		}
		std::printf("Kyty pilote interne : tenir %u ms, %u retours sur la cible\n", ms, backs);
		std::fflush(stdout);
		return true;
	}

	// v4 (20/11, "TAS" replay): "rejouer <commandes.csv> <image de debut> <image de fin> [tolerance=1.5]".
	// commandes.csv (journal v4) holds every pad state the game read during a session, with the guest
	// flip index and the camera. The replay applies them frame by frame on the guest flip clock (a slower
	// game gets the same inputs per frame), from the frame the step starts at. The emulator is not
	// deterministic (loading times, enemies), so every 10 frames the camera is compared with the recorded
	// one: past the tolerance, the driver walks back to the recorded place ("aller") and resumes from the
	// recorded frame nearest to where Astro now is, within the next 10 seconds of the recording.
	struct Recorded {
		uint64_t image = 0;
		uint32_t buttons = 0;
		uint8_t  axes[6] {};
		float    x = 0, z = 0;
		bool     has_camera = false;
	};

	static std::vector<Recorded> LoadRecording(const std::string& path, uint64_t from, uint64_t to) {
		std::vector<Recorded> out;
		std::ifstream         in(path, std::ios::binary);
		std::string           line;
		while (std::getline(in, line)) {
			// image;t_s;heure;boutons;lx;ly;rx;ry;l2;r2;camera
			std::vector<std::string> f;
			std::string              cur;
			for (char c: line) {
				if (c == ';') {
					f.push_back(cur);
					cur.clear();
				} else if (c != '\r') {
					cur += c;
				}
			}
			f.push_back(cur);
			if (f.size() < 10 || f[0].empty() || !std::isdigit(static_cast<unsigned char>(f[0][0]))) {
				continue;
			}
			Recorded r;
			r.image = std::strtoull(f[0].c_str(), nullptr, 10);
			if (r.image > to) {
				break;
			}
			r.buttons = static_cast<uint32_t>(std::strtoul(f[3].c_str(), nullptr, 16));
			for (int k = 0; k < 6; k++) {
				r.axes[k] = static_cast<uint8_t>(std::clamp(std::atoi(f[4 + k].c_str()), 0, 255));
			}
			if (f.size() > 10) {
				float y = 0;
				std::istringstream cam(f[10]);
				r.has_camera = static_cast<bool>(cam >> r.x >> y >> r.z);
			}
			if (r.image >= from) {
				out.push_back(r);
			} else {
				// the state in force at `from`: the last one before it
				if (out.empty()) {
					out.push_back(r);
				} else {
					out[0] = r;
				}
				out[0].image = from;
			}
		}
		return out;
	}

	static void ApplyRecorded(const Recorded& r) {
		static constexpr uint32_t BUTTONS[] = {
		    Controller::PAD_BUTTON_CROSS, Controller::PAD_BUTTON_SQUARE, Controller::PAD_BUTTON_TRIANGLE,
		    Controller::PAD_BUTTON_CIRCLE, Controller::PAD_BUTTON_L1, Controller::PAD_BUTTON_R1,
		    Controller::PAD_BUTTON_L2, Controller::PAD_BUTTON_R2, Controller::PAD_BUTTON_OPTIONS,
		    0x00000002u /* L3 */, 0x00000004u /* R3 */, 0x00000010u, 0x00000020u, 0x00000040u, 0x00000080u /* pad */};
		const int id = Controller::HOST_INPUT_CONTROLLER_ID;
		for (const auto b: BUTTONS) {
			Controller::SetButton(id, b, (r.buttons & b) != 0);
		}
		Controller::SetAxis(id, Controller::Axis::LeftX, r.axes[0]);
		Controller::SetAxis(id, Controller::Axis::LeftY, r.axes[1]);
		Controller::SetAxis(id, Controller::Axis::RightX, r.axes[2]);
		Controller::SetAxis(id, Controller::Axis::RightY, r.axes[3]);
		Controller::SetAxis(id, Controller::Axis::TriggerLeft, r.axes[4]);
		Controller::SetAxis(id, Controller::Axis::TriggerRight, r.axes[5]);
	}

	// "rejouer": false if the file changed or the driver stops.
	bool Replay(const Step& s) {
		const auto rec = LoadRecording(s.file, s.from, s.to);
		if (rec.empty()) {
			std::printf("Kyty pilote interne : rejouer : aucune commande entre les images %llu et %llu de %s\n",
			            static_cast<unsigned long long>(s.from), static_cast<unsigned long long>(s.to), s.file.c_str());
			std::fflush(stdout);
			return true;
		}
		m_active          = true;
		int64_t  offset   = static_cast<int64_t>(Libs::Graphics::FluidityFlipIndex()) - static_cast<int64_t>(s.from);
		size_t   k        = 0, applied = SIZE_MAX;
		uint64_t checked  = 0;
		uint32_t resyncs  = 0;
		float    worst    = 0;
		std::printf("Kyty pilote interne : rejouer : %zu etats, images %llu a %llu\n", rec.size(),
		            static_cast<unsigned long long>(s.from), static_cast<unsigned long long>(s.to));
		std::fflush(stdout);
		for (;;) {
			if (m_stop || Changed()) {
				m_reload = !m_stop;
				return false;
			}
			const auto f = static_cast<uint64_t>(std::max<int64_t>(
			    0, static_cast<int64_t>(Libs::Graphics::FluidityFlipIndex()) - offset));
			if (f > s.to) {
				break;
			}
			while (k + 1 < rec.size() && rec[k + 1].image <= f) {
				k++;
			}
			if (k != applied) {
				ApplyRecorded(rec[k]);
				applied = k;
			}
			// Position check every 10 frames, against the last recorded camera at or before this frame.
			if (f >= checked + 10) {
				checked = f;
				size_t c = k;
				while (c > 0 && !rec[c].has_camera) {
					c--;
				}
				float pos[3] {}, dir[3] {};
				if (rec[c].has_camera && Libs::Graphics::FluidityCamera(pos, dir)) {
					const float d = std::hypot(rec[c].x - pos[0], rec[c].z - pos[2]);
					worst         = std::max(worst, d);
					if (d > s.tolerance) {
						// Walk back to the recorded place, then resume from the nearest recorded frame ahead.
						Step back {};
						back.kind      = 2;
						back.x         = rec[c].x;
						back.z         = rec[c].z;
						back.tolerance = 0.5f;
						back.ms        = 6000;
						Recorded neutral {};
						std::fill(std::begin(neutral.axes), std::begin(neutral.axes) + 4, uint8_t {128});
						ApplyRecorded(neutral);
						m_quiet       = true;
						const bool ok = Go(back);
						m_quiet       = false;
						if (!ok) {
							return false;
						}
						resyncs++;
						float p2[3] {}, d2[3] {};
						size_t best = c;
						if (Libs::Graphics::FluidityCamera(p2, d2)) {
							float best_d = 1e30f;
							for (size_t q = c; q < rec.size() && rec[q].image <= rec[c].image + 300; q++) {
								if (!rec[q].has_camera) {
									continue;
								}
								const float dq = std::hypot(rec[q].x - p2[0], rec[q].z - p2[2]);
								if (dq < best_d) {
									best_d = dq;
									best   = q;
								}
							}
						}
						k       = best;
						applied = SIZE_MAX;
						offset  = static_cast<int64_t>(Libs::Graphics::FluidityFlipIndex()) -
						         static_cast<int64_t>(rec[best].image);
						checked = rec[best].image;
						continue;
					}
				}
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
		}
		Recorded neutral {};
		std::fill(std::begin(neutral.axes), std::begin(neutral.axes) + 4, uint8_t {128});
		ApplyRecorded(neutral);
		float p2[3] {}, d2[3] {};
		const bool cam = Libs::Graphics::FluidityCamera(p2, d2);
		std::printf("Kyty pilote interne : rejouer : fini, %u recalages, ecart max %.2f, camera %.3f %.3f %.3f\n", resyncs,
		            worst, cam ? p2[0] : 0.0f, cam ? p2[1] : 0.0f, cam ? p2[2] : 0.0f);
		std::fflush(stdout);
		return true;
	}

	static constexpr const char* KEYS[] = {"W", "A", "S", "D", "F", "H", "T", "G", "J", "K", "I", "L",
	                                       "Q", "E", "Z", "C", "ENTREE"};

	static int KeyIndex(const std::string& k) {
		for (int i = 0; i < static_cast<int>(std::size(KEYS)); i++) {
			if (k == KEYS[i]) {
				return i;
			}
		}
		return -1;
	}

	static void Apply(uint32_t keys) {
		const auto on = [keys](int i) { return (keys >> i) & 1u; };
		const auto axis = [](bool neg, bool pos) { return neg && !pos ? 0 : (pos && !neg ? 255 : 128); };
		const int  id   = Controller::HOST_INPUT_CONTROLLER_ID;
		Controller::SetAxis(id, Controller::Axis::LeftX, axis(on(1), on(3)));
		Controller::SetAxis(id, Controller::Axis::LeftY, axis(on(0), on(2)));
		Controller::SetAxis(id, Controller::Axis::RightX, axis(on(4), on(5)));
		Controller::SetAxis(id, Controller::Axis::RightY, axis(on(6), on(7)));
		static constexpr uint32_t BUTTONS[] = {Controller::PAD_BUTTON_CROSS,    Controller::PAD_BUTTON_SQUARE,
		                                       Controller::PAD_BUTTON_TRIANGLE, Controller::PAD_BUTTON_CIRCLE,
		                                       Controller::PAD_BUTTON_L1,       Controller::PAD_BUTTON_R1,
		                                       Controller::PAD_BUTTON_L2,       Controller::PAD_BUTTON_R2,
		                                       Controller::PAD_BUTTON_OPTIONS};
		for (int b = 0; b < static_cast<int>(std::size(BUTTONS)); b++) {
			Controller::SetButton(id, BUTTONS[b], on(8 + b) != 0);
		}
	}

	void ReleaseAll() {
		m_presses.clear();
		m_button_keys = 0;
		if (m_active) {
			Apply(0);
			m_active = false;
		}
	}

	bool Changed() {
		std::error_code ec;
		const auto      t    = std::filesystem::last_write_time(m_path, ec);
		const auto      size = ec ? uintmax_t {0} : std::filesystem::file_size(m_path, ec);
		const auto      stamp = ec ? 0 : static_cast<int64_t>(t.time_since_epoch().count()) ^ static_cast<int64_t>(size);
		if (stamp == m_stamp) {
			return false;
		}
		m_stamp = stamp;
		return true;
	}

	void Load() {
		m_steps.clear();
		m_loop = SIZE_MAX; // no "boucle": run once
		std::ifstream in(m_path, std::ios::binary);
		std::string   text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		for (auto& c : text) {
			if (c == ';' || c == '\r' || c == '\n') {
				c = '\n';
			}
		}
		std::istringstream lines(text);
		std::string        line;
		while (std::getline(lines, line)) {
			std::istringstream words(line);
			std::string        verb, key;
			words >> verb;
			if (verb.empty()) {
				continue;
			}
			std::transform(verb.begin(), verb.end(), verb.begin(), [](unsigned char c) { return std::tolower(c); });
			if (verb == "boucle") {
				m_loop = m_steps.size();
				continue;
			}
			Step s;
			uint32_t ms = 0;
			if (verb == "attendre") {
				words >> ms;
				s.ms = ms;
			} else if (verb == "maintenir" || verb == "touche") {
				words >> key;
				ms = verb == "touche" ? 120 : 0;
				words >> ms;
				std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::toupper(c); });
				std::istringstream parts(key);
				std::string        part;
				while (std::getline(parts, part, '+')) {
					const int i = KeyIndex(part);
					if (i >= 0) {
						s.keys |= 1u << i;
					}
				}
				s.kind = 1;
				s.ms   = ms;
			} else if (verb == "tenir") {
				words >> ms;
				s.kind = 3;
				s.ms   = ms;
				m_steps.push_back(s);
				continue;
			} else if (verb == "attendre-calme") {
				uint32_t still = 2000, max_ms = 300000;
				words >> still >> max_ms;
				s.kind   = 7;
				s.ms     = still;
				s.max_ms = max_ms;
				m_steps.push_back(s);
				continue;
			} else if (verb == "rejoindre") {
				s.kind = 8;
				m_steps.push_back(s);
				continue;
			} else if (verb == "bouton") {
				words >> key >> ms;
				std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::toupper(c); });
				std::istringstream parts(key);
				std::string        part;
				while (std::getline(parts, part, '+')) {
					const int i = KeyIndex(part);
					if (i >= 8) {
						s.keys |= 1u << i;
					}
				}
				s.kind = 5;
				s.ms   = ms;
				m_steps.push_back(s);
				continue;
			} else if (verb == "passer" || verb == "attendre-pres") {
				const bool passing = verb == "passer";
				float      x = 0, z = 0, tolerance = passing ? 0.6f : 1.0f;
				uint32_t   max_ms = passing ? 8000 : 300000;
				if (!(words >> x >> z)) {
					continue;
				}
				words >> tolerance >> max_ms;
				s.kind      = passing ? 4 : 6;
				s.x         = x;
				s.z         = z;
				s.tolerance = tolerance > 0 ? tolerance : 0.6f;
				s.ms        = max_ms;
				m_steps.push_back(s);
				continue;
			} else if (verb == "rejouer") {
				std::string file;
				uint64_t    from = 0, to = 0;
				float       tolerance = 1.5f;
				if (!(words >> file >> from >> to)) {
					continue;
				}
				words >> tolerance;
				s.kind      = 9;
				s.file      = file;
				s.from      = from;
				s.to        = to;
				s.tolerance = tolerance > 0 ? tolerance : 1.5f;
				m_steps.push_back(s);
				continue;
			} else if (verb == "aller") {
				float    x = 0, z = 0, tolerance = 0.05f;
				uint32_t max_ms = 20000;
				if (!(words >> x >> z)) {
					continue;
				}
				words >> tolerance >> max_ms;
				s.kind      = 2;
				s.x         = x;
				s.z         = z;
				s.tolerance = tolerance > 0 ? tolerance : 0.05f;
				s.ms        = max_ms;
				m_steps.push_back(s);
				continue;
			} else {
				continue;
			}
			m_steps.push_back(s);
			if (s.kind == 1) {
				m_steps.push_back(Step {0, 0, 30}); // short release between holds
			}
		}
		std::printf("Kyty pilote interne : %zu etapes, boucle a partir de %zd\n", m_steps.size(),
		            m_loop == SIZE_MAX ? static_cast<ptrdiff_t>(-1) : static_cast<ptrdiff_t>(m_loop));
		std::fflush(stdout);
	}

	// Waits `ms`, false if the file changed or the driver stops.
	bool Sleep(uint32_t ms) {
		const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
		while (std::chrono::steady_clock::now() < end) {
			if (m_stop) {
				return false;
			}
			if (Changed()) {
				m_reload = true;
				return false;
			}
			if (!m_presses.empty() || m_button_keys != 0) {
				UpdateButtons();
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
		}
		return true;
	}

	void Run() {
		m_stamp  = -1;
		m_reload = Changed() || true;
		while (!m_stop) {
			if (m_reload) {
				m_reload = false;
				ReleaseAll();
				Load();
			}
			if (m_steps.empty()) {
				Sleep(200);
				continue;
			}
			std::size_t i = 0;
			while (!m_stop && !m_reload) {
				if (i >= m_steps.size()) {
					if (m_loop >= m_steps.size()) {
						ReleaseAll();
						m_steps.clear(); // finished, wait for a new script
						break;
					}
					i = m_loop;
				}
				const auto& s = m_steps[i++];
				if (s.kind == 5) {
					m_presses.push_back({s.keys, std::chrono::steady_clock::now() + std::chrono::milliseconds(s.ms)});
					UpdateButtons();
					continue;
				}
				if (s.kind == 6) {
					if (!WaitNear(s)) {
						break;
					}
					continue;
				}
				if (s.kind == 7) {
					if (!WaitCalm(s)) {
						break;
					}
					continue;
				}
				if (s.kind == 8) {
					i = Join(i);
					continue;
				}
				if (s.kind == 9) {
					if (!Replay(s)) {
						break;
					}
					continue;
				}
				if (s.kind == 2 || s.kind == 3 || s.kind == 4) {
					if (!(s.kind == 3 ? Hold(s.ms) : Go(s))) {
						break;
					}
					size_t next = i;
					while (next < m_steps.size() && m_steps[next].kind == 5) {
						next++; // presses between waypoints keep the run going
					}
					if (s.kind == 4 && (next >= m_steps.size() || m_steps[next].kind != 4)) {
						// end of a run of waypoints: stop, and say how it went
						ApplyStick(0, 0);
						float p2[3] {}, d2[3] {};
						if (Libs::Graphics::FluidityCamera(p2, d2)) {
							std::printf("Kyty pilote interne : trajet : %u points passes, %u delais, camera %.3f %.3f %.3f\n",
							            m_passed, m_timeouts, p2[0], p2[1], p2[2]);
							std::fflush(stdout);
						}
					}
					continue;
				}
				Apply(s.kind == 1 ? s.keys : 0);
				m_active = true;
				if (!Sleep(s.ms)) {
					break;
				}
			}
		}
		ReleaseAll();
	}

	std::string       m_path;
	std::thread       m_thread;
	std::atomic<bool> m_stop {false};
	bool              m_reload = false;
	bool              m_active = false;
	int64_t           m_stamp  = -1;
	std::vector<Step> m_steps;
	std::size_t       m_loop = 0;
	int               m_side = 0; // "aller": stick side, -1 (KYTY_PAD_SIDE=1: +1)
	Step              m_target {};
	bool              m_has_target = false;
	bool              m_quiet      = false;
};

PadScript g_pad_script;

} // namespace

void HostInputInit(SDL_Window* window) {
	g_pad_script.Start(); // KYTY_PAD_FILE
	GetInputMap();
	g_mouse_window = window;
	if (const char* value = std::getenv("KYTY_HOST_INPUT_MIN_PRESS_MS")) {
		char* end = nullptr;
		const long ms = std::strtol(value, &end, 10);
		if (end != value && *end == '\0' && ms >= 0 && ms <= 2000) {
			g_input_pulse.SetMinimum(static_cast<uint64_t>(ms));
			LOGF("Host input minimum press: %ld ms (test mode)\n", ms);
		}
	}
	if (Config::HideCursorEnabled()) {
		g_cursor_hide_at = SDL_GetTicks() + CURSOR_IDLE_HIDE_MS;
	}
}

void HostInputShutdown() {
	g_pad_script.Stop(); // KYTY_PAD_FILE
	g_input_pulse.ReleaseAll(ApplyHostKey);
	g_pad_trigger_pressed.clear();
	HostInputSetTouchPadMode(false);
	int gamepad_touch_id = -1;
	{
		std::scoped_lock lock(g_touchpad_mutex);
		gamepad_touch_id = g_gamepad_touchpad.enabled ? g_gamepad_touchpad.id : -1;
	}
	if (gamepad_touch_id >= 0) {
		HostInputDisconnectGamepadTouchPad(gamepad_touch_id);
	}
	if (g_mouse.enabled) {
		SetRelativeMouseMode(false);
		CenterMouseStick();
		g_mouse = {};
	}
	g_mouse_window   = nullptr;
	g_cursor_hide_at = 0;
	if (Config::HideCursorEnabled()) {
		SDL_ShowCursor();
	}
}

void HostInputKey(int key_code, bool down) {
	Common::HostInputTrace("host-key", key_code, down);
	g_input_pulse.Key(key_code, down, SDL_GetTicks(), ApplyHostKey);
}

void HostInputMouseButton(uint8_t mouse_button, bool down) {
	const auto& map = GetInputMap();
	if (map.Custom() && mouse_button != 0) {
		SetControl(map.FindMouseButton(mouse_button), down);
	}
}

void HostInputMouseTouchPad(bool down, float x, float y) {
	RecordMouseTouchPadPosition(x, y);
	{
		std::scoped_lock lock(g_touchpad_mutex);
		if (!g_mouse_touchpad.enabled) {
			return;
		}
		if (down && !g_mouse_touchpad.drawing) {
			g_mouse_touchpad.gesture++;
		}
		g_mouse_touchpad.drawing = down;
		g_mouse_touchpad.revision++;
		x = g_mouse_touchpad.x;
		y = g_mouse_touchpad.y;
	}
	SetSwipeTouchPad(down, x, y);
}

void HostInputMouseMotion(float x, float y) {
	RecordMouseTouchPadPosition(x, y);
	std::scoped_lock lock(g_touchpad_mutex);
	if (g_mouse_touchpad.enabled && g_mouse_touchpad.drawing) {
		SetSwipeTouchPad(true, g_mouse_touchpad.x, g_mouse_touchpad.y);
	}
}

void HostInputSetTouchPadMode(bool enabled) {
	if (enabled) {
		int gamepad_id = -1;
		{
			std::scoped_lock lock(g_touchpad_mutex);
			if (g_gamepad_touchpad.enabled) {
				gamepad_id = g_gamepad_touchpad.id;
			}
		}
		if (gamepad_id >= 0) {
			HostInputDisconnectGamepadTouchPad(gamepad_id);
		}
	}
	{
		std::scoped_lock lock(g_touchpad_mutex);
		if (g_mouse_touchpad.enabled == enabled) {
			return;
		}
	}
	if (!enabled) {
		float x = 0;
		float y = 0;
		(void) SDL_GetMouseState(&x, &y);
		HostInputMouseTouchPad(false, x, y);
	}
	bool show_system_cursor = false;
	{
		std::scoped_lock lock(g_touchpad_mutex);
		g_mouse_touchpad.enabled        = enabled;
		g_mouse_touchpad.cursor_visible = true;
		g_mouse_touchpad.drawing        = false;
		g_mouse_touchpad.position_valid = false;
		g_mouse_touchpad.gesture++;
		g_mouse_touchpad.revision++;
		show_system_cursor = !enabled && !g_gamepad_touchpad.enabled;
	}
	if (enabled) {
		if (!SDL_HideCursor()) {
			LOGF("Failed to hide mouse cursor for touchpad mode: %s\n", SDL_GetError());
		}
	} else if (show_system_cursor && !Config::HideCursorEnabled() && !SDL_ShowCursor()) {
		LOGF("Failed to restore mouse cursor after touchpad mode: %s\n", SDL_GetError());
	}
	LOGF("Mouse touchpad mode: %s\n", enabled ? "enabled" : "disabled");
}

void HostInputToggleTouchPadMode() {
	bool enabled = false;
	{
		std::scoped_lock lock(g_touchpad_mutex);
		enabled = !g_mouse_touchpad.enabled;
	}
	HostInputSetTouchPadMode(enabled);
}

void HostInputToggleGamepadTouchPad(int id) {
	bool mouse_mode = false;
	{
		std::scoped_lock lock(g_touchpad_mutex);
		mouse_mode = g_mouse_touchpad.enabled;
	}
	if (mouse_mode) {
		HostInputSetTouchPadMode(false);
	}

	const auto stick = GamepadLeftStick(id);
	bool  enabled = false;
	float x       = 0.5f;
	float y       = 0.5f;
	{
		std::scoped_lock lock(g_touchpad_mutex);
		if (g_gamepad_touchpad.enabled && g_gamepad_touchpad.id != id) {
			return;
		}
		enabled = !g_gamepad_touchpad.enabled;
		if (enabled) {
			g_gamepad_touchpad           = {};
			g_gamepad_touchpad.enabled   = true;
			g_gamepad_touchpad.id        = id;
			g_gamepad_touchpad.x_axis    = stick[0];
			g_gamepad_touchpad.y_axis    = stick[1];
			g_gamepad_touchpad.last_poll = SDL_GetTicks();
			g_gamepad_touchpad.next_poll = g_gamepad_touchpad.last_poll + GAMEPAD_TOUCHPAD_POLL_MS;
			g_mouse_touchpad.gesture++;
		} else {
			x                          = g_gamepad_touchpad.x;
			y                          = g_gamepad_touchpad.y;
			g_gamepad_touchpad         = {};
		}
		g_mouse_touchpad.revision++;
	}
	Controller::SetAxis(id, Controller::Axis::LeftX, enabled ? 128 : stick[0]);
	Controller::SetAxis(id, Controller::Axis::LeftY, enabled ? 128 : stick[1]);
	if (enabled) {
		if (!SDL_HideCursor()) {
			LOGF("Failed to hide mouse cursor for gamepad touchpad mode: %s\n", SDL_GetError());
		}
	} else {
		Controller::SetTouchPad(id, 1, false, x, y);
		if (!Config::HideCursorEnabled() && !SDL_ShowCursor()) {
			LOGF("Failed to restore mouse cursor after gamepad touchpad mode: %s\n",
			     SDL_GetError());
		}
	}
	LOGF("Gamepad touchpad mode: %s\n", enabled ? "enabled" : "disabled");
}

bool HostInputGamepadDrawButton(int id, bool down) {
	float x = 0;
	float y = 0;
	{
		std::scoped_lock lock(g_touchpad_mutex);
		if (!g_gamepad_touchpad.enabled || g_gamepad_touchpad.id != id) {
			return false;
		}
		if (!down && !g_gamepad_touchpad.drawing) {
			return false;
		}
		if (down && !g_gamepad_touchpad.drawing) {
			g_mouse_touchpad.gesture++;
		}
		g_gamepad_touchpad.drawing = down;
		x                          = g_gamepad_touchpad.x;
		y                          = g_gamepad_touchpad.y;
		g_mouse_touchpad.revision++;
	}
	Controller::SetTouchPad(id, 1, down, x, y);
	return true;
}

bool HostInputGamepadAxis(int id, bool horizontal, int value) {
	std::scoped_lock lock(g_touchpad_mutex);
	if (!g_gamepad_touchpad.enabled || g_gamepad_touchpad.id != id) {
		return false;
	}
	(horizontal ? g_gamepad_touchpad.x_axis : g_gamepad_touchpad.y_axis) = std::clamp(value, 0, 255);
	g_gamepad_touchpad.next_poll = SDL_GetTicks();
	return true;
}

void HostInputDisconnectGamepadTouchPad(int id) {
	float x = 0;
	float y = 0;
	{
		std::scoped_lock lock(g_touchpad_mutex);
		if (!g_gamepad_touchpad.enabled || g_gamepad_touchpad.id != id) {
			return;
		}
		x                  = g_gamepad_touchpad.x;
		y                  = g_gamepad_touchpad.y;
		g_gamepad_touchpad = {};
		g_mouse_touchpad.revision++;
	}
	Controller::SetTouchPad(id, 1, false, x, y);
	const auto stick = GamepadLeftStick(id);
	Controller::SetAxis(id, Controller::Axis::LeftX, stick[0]);
	Controller::SetAxis(id, Controller::Axis::LeftY, stick[1]);
	if (!TouchPadModeActive() && !Config::HideCursorEnabled() && !SDL_ShowCursor()) {
		LOGF("Failed to restore mouse cursor after gamepad disconnect: %s\n", SDL_GetError());
	}
}

HostInputTouchPadVisualState GetHostInputTouchPadVisualState() {
	std::scoped_lock             lock(g_touchpad_mutex);
	HostInputTouchPadVisualState state {};
	if (g_gamepad_touchpad.enabled) {
		state.enabled        = true;
		state.cursor_visible = true;
		state.drawing        = g_gamepad_touchpad.drawing;
		state.x              = g_gamepad_touchpad.x;
		state.y              = g_gamepad_touchpad.y;
		state.revision       = g_mouse_touchpad.revision;
		state.gesture        = g_mouse_touchpad.gesture;
		state.position_valid = true;
		return state;
	}
	state.enabled        = g_mouse_touchpad.enabled;
	state.cursor_visible = g_mouse_touchpad.cursor_visible;
	state.drawing        = g_mouse_touchpad.drawing;
	state.x              = g_mouse_touchpad.x;
	state.y              = g_mouse_touchpad.y;
	state.revision       = g_mouse_touchpad.revision;
	state.gesture        = g_mouse_touchpad.gesture;
	state.position_valid = g_mouse_touchpad.position_valid;
	return state;
}

void HostInputToggleMouseToJoystick() {
	if (g_mouse.enabled) {
		SetRelativeMouseMode(false);
		CenterMouseStick();
		g_mouse = {};
		LOGF("Mouse to right stick: disabled\n");
		return;
	}

	if (!SetRelativeMouseMode(true)) {
		return;
	}
	g_mouse.enabled = true;
	LOGF("Mouse to right stick: enabled (F7 to release)\n");
}

namespace {
bool g_input_suspended = false; // fil de la fenetre seulement
} // namespace

void HostInputSetSuspended(bool suspended) {
	if (g_input_suspended == suspended) {
		return;
	}
	g_input_suspended = suspended;
	if (suspended) {
		g_input_pulse.ReleaseAll(ApplyHostKey);
		HostInputSetTouchPadMode(false);
		int gamepad_touch_id = -1;
		{
			std::scoped_lock lock(g_touchpad_mutex);
			gamepad_touch_id = g_gamepad_touchpad.enabled ? g_gamepad_touchpad.id : -1;
		}
		if (gamepad_touch_id >= 0) {
			HostInputDisconnectGamepadTouchPad(gamepad_touch_id);
		}
		if (g_mouse.enabled) {
			SetRelativeMouseMode(false);
			g_mouse.next_poll = 0;
			CenterMouseStick();
		}
		SDL_ShowCursor();
	} else {
		if (g_mouse.enabled) {
			SetRelativeMouseMode(true);
			g_mouse.next_poll = 0;
		}
		if (Config::HideCursorEnabled()) {
			g_cursor_hide_at = SDL_GetTicks() + CURSOR_IDLE_HIDE_MS;
		}
	}
}

bool HostInputWaitEvent(SDL_Event* event, int max_wait_ms) {
	// Astro Bot PC (13/11) : voir Fsr3::HoldWindowPump.
	if (Fsr3::HoldWindowPump()) {
		SDL_Delay(1);
		return false;
	}
	// -1 waits without a deadline (until max_wait_ms below).
	int timeout = g_input_pulse.Poll(SDL_GetTicks(), ApplyHostKey);
	if (!g_mouse.enabled || g_input_suspended || SDL_GetKeyboardFocus() != g_mouse_window) {
		g_mouse.next_poll = 0;
		CenterMouseStick();
	} else {
		if (g_mouse.next_poll == 0) {
			SDL_GetRelativeMouseState(nullptr, nullptr);
			g_mouse.next_poll = SDL_GetTicks() + MOUSE_POLL_INTERVAL_MS;
		}
		const int mouse_wait = PollMouse(SDL_GetTicks());
		timeout              = timeout < 0 ? mouse_wait : std::min(timeout, mouse_wait);
	}

	if (g_cursor_hide_at != 0 && !TouchPadModeActive() && !g_input_suspended) {
		const auto now_ms = SDL_GetTicks();
		const int  cursor_timeout =
		    now_ms < g_cursor_hide_at ? static_cast<int>(g_cursor_hide_at - now_ms) : 0;
		timeout = timeout < 0 ? cursor_timeout : std::min(timeout, cursor_timeout);
	}
	if (const int touch_timeout = PollGamepadTouchPad(SDL_GetTicks()); touch_timeout >= 0) {
		timeout = timeout < 0 ? touch_timeout : std::min(timeout, touch_timeout);
	}
	timeout              = timeout < 0 ? max_wait_ms : std::min(timeout, max_wait_ms);
	const bool has_event = SDL_WaitEventTimeout(event, timeout);
	if (!has_event && timeout < 0) {
		EXIT("%s\n", SDL_GetError());
	}

	if (Config::HideCursorEnabled() && !TouchPadModeActive() && !g_input_suspended) {
		const auto now_ms = SDL_GetTicks();
		if (has_event && !g_mouse.enabled && IsCursorActivity(*event) &&
		    SDL_GetWindowFromEvent(event) == g_mouse_window) {
			SDL_ShowCursor();
			g_cursor_hide_at = now_ms + CURSOR_IDLE_HIDE_MS;
		} else if (g_cursor_hide_at != 0 && now_ms >= g_cursor_hide_at) {
			SDL_HideCursor();
			g_cursor_hide_at = 0;
		}
	}

	if (has_event && event->type == SDL_EVENT_WINDOW_FOCUS_LOST &&
	    event->window.windowID == SDL_GetWindowID(g_mouse_window)) {
		Common::HostInputTrace("focus-lost", 0, 0);
		g_input_pulse.ReleaseAll(ApplyHostKey);
		g_mouse.next_poll = 0;
		CenterMouseStick();
	}
	return has_event;
}

} // namespace Libs::Graphics
