
from typing import Any, Callable, List, Literal, NamedTuple


class Selection(NamedTuple):
  context: List[Any]
  future: List[Any]


def dino_foresight_eval_selection(
    context_length: int,
    horizon: Literal["short", "medium", "long", "uncertain"],
    benchmark: Literal["kubric", "cityscapes"]
) -> Callable[[List[Any]], Selection]:
  sequence_length = context_length + 1

  match horizon:
    case "short":
      num_frames_skip = 2
      step = num_frames_skip + 1
      start_idx = 20 - step * sequence_length + num_frames_skip
      gt_idx = 19
    case "medium":
      num_frames_skip = 2
      step = num_frames_skip + 1
      start_idx = 20 - step * sequence_length + num_frames_skip - 6
      gt_idx = 19
    case "long":
      num_frames_skip = 1
      step = num_frames_skip + 1
      gt_idx = 23
      start_idx = (
          (gt_idx + 1) - step * sequence_length + num_frames_skip - 7 * 2
      )
    case "uncertain":
      # step = 1
      # gt_idx = 23
      # [5, 7, 9, 11], [13, 15, 17, 19]
      match benchmark:
        case "kubric":
          num_frames_skip = 1
          step = num_frames_skip + 1
          start_idx = 0
          # [0, 2, 4, 6] [8, 10, 12, 14, 16, 18, 20, 22]
          gt_idx = 22
        case _:
          num_frames_skip = 1
          step = num_frames_skip + 1
          start_idx = 1
          gt_idx = 19

  def selector(xs: List[Any]) -> Selection:
    if horizon == "vae" or horizon == "full":
      return Selection(
          context=xs,
          future=xs
      )
    else:
      return Selection(
          context=xs[start_idx: start_idx + step * (context_length): step],
          future=xs[start_idx + step * context_length: gt_idx + 1: step]
      )
  return selector
