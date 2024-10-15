import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()  # 所有对话内容会被 sep 分隔
    TWO = auto()  # 使用两个分隔符 sep 和 sep2 交替地将消息分开
    MPT = auto()  # 这里会直接使用 sep 分隔每条消息
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        """
        生成对话完整文本的核心方法，返回一个字符串表示整个对话。方法的逻辑会根据 sep_style 选择不同的格式来生成输出。
        """
        messages = self.messages
        # 这里。message的大致格式应为：("role1", ("message1", "additional_info"))
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()  # 避免修改原始消息
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<graph>", "").strip()  # 将第一条消息的<graph>去掉，去除首尾空白
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Graph><graph></Graph>"))
                messages.insert(1, (self.roles[1], "Received."))
                # self.version = "v0_mmtag"
                # self.roles = ["Human", "Assistant"]
                # self.messages = [
                #     ("Human", ("<graph>Some graph content", "additional_info")),
                #     ("Assistant", "Some response")
                # ]

                # messages = [
                #     ("Human", "<Graph><graph></Graph>"),
                #     ("Assistant", "Received."),
                #     ("Human", "Some graph content")
                # ]
            else:
                messages[0] = (init_role, "<graph>\n" + init_msg)
                # messages = [
                #     ("Human", "<graph>\nSome graph content")
                # ]
            if self.sep_style == SeparatorStyle.SINGLE:
                ret = self.system + self.sep
                for role, message in messages:
                    if message:
                        if type(message) is tuple:
                            message, _, _ = message
                        ret += role + ": " + message + self.sep
                    else:
                        ret += role + ":"
            elif self.sep_style == SeparatorStyle.TWO:
                seps = [self.sep, self.sep2]
                ret = self.system + seps[0]
                for i, (role, message) in enumerate(messages):
                    if message:
                        if type(message) is tuple:
                            message, _, _ = message
                        ret += role + ": " + message + seps[i % 2]
                    else:
                        ret += role + ":"
            elif self.sep_style == SeparatorStyle.MPT:
                ret = self.system + self.sep
                for role, message in messages:
                    if message:
                        if type(message) is tuple:
                            message, _, _ = message
                        ret += role + message + self.sep
                    else:
                        ret += role
            elif self.sep_style == SeparatorStyle.LLAMA_2:
                wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
                wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
                ret = ""

                for i, (role, message) in enumerate(messages):
                    if i == 0:
                        assert message, "first message should not be none"
                        assert role == self.roles[0], "first message should come from user"
                    if message:
                        if type(message) is tuple:
                            message, _, _ = message
                        if i == 0: message = wrap_sys(self.system) + message
                        if i % 2 == 0:
                            message = wrap_inst(message)
                            ret += self.sep + message
                        else:
                            ret += " " + message + " " + self.sep2
                    else:
                        ret += ""
                ret = ret.lstrip(self.sep)
            elif self.sep_style == SeparatorStyle.PLAIN:
                seps = [self.sep, self.sep2]
                ret = self.system
                for i, (role, message) in enumerate(messages):
                    if message:
                        if type(message) is tuple:
                            message, _, _ = message
                        ret += message + seps[i % 2]
                    else:
                        ret += ""
            else:
                raise ValueError(f"Invalid style: {self.sep_style}")
            return ret

        def append_message(self, role, message):
            self.messages.append([role, message])

        def copy(self):
            """
            复制当前的对话对象
            """
            return Conversation(
                system=self.system,
                roles=self.roles,
                messages=[[x, y] for x, y in self.messages],
                offset=self.offset,
                sep_style=self.sep_style,
                sep=self.sep,
                sep2=self.sep2,
                version=self.version)

        def dict(self):
            """
            将当前对话对象转换为字典形式
            """
            # if len(self.get_images()) > 0:
            #     return {
            #         "system": self.system,
            #         "roles": self.roles,
            #         "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
            #         "offset": self.offset,
            #         "sep": self.sep,
            #         "sep2": self.sep2,
            #     }
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": self.messages,
                "offset": self.offset,  # 控制生成对话时从哪里开始
                "sep": self.sep,
                "sep2": self.sep2,
            }

conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llaga_llama_2 = Conversation(
    system="You are a helpful language and graph assistant. "
           "You are able to understand the graph content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

# conv_mpt = Conversation(
#     system="""<|im_start|>system
# A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
#     roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
#     version="mpt",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.MPT,
#     sep="<|im_end|>",
# )

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="</s>",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the graph content that the user provides, and assist the user with a variety of tasks using natural language."
           "The graph content will be provided with the following format: <Graph>graph content</Graph>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the graph content that the user provides, and assist the user with a variety of tasks using natural language."
           "The graph content will be provided with the following format: <Graph>graph content</Graph>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

default_conversation = conv_vicuna_v0
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "llaga_llama_2": conv_llaga_llama_2,
    "mpt": conv_mpt,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())