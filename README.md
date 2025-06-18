# The Multi-session Task-oriented Dialogue Dataset (MS-TOD) Dataset

*   [Paper - MS-TOD dataset and Memory-Active Policy]([https://arxiv.org/pdf/1909.05855.pdf](https://arxiv.org/abs/2505.20231))
*   [Paper - SGD dataset](https://arxiv.org/pdf/1909.05855.pdf)
*   [GitHub - SGD dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

## Overview

Existing Task-Oriented Dialogue (TOD) systems primarily focus on single-session dialogues, limiting their effectiveness in long-term memory augmentation. To address this challenge, we introduce the **Multi-session Task-oriented Dialogue Dataset (MS-TOD)** dataset. Derived from the **Schema-Guided Dialogue (SGD)** dataset, it is the first multi-session TOD dataset designed to retain long-term memory across sessions, enabling fewer turns and more efficient task completion. This defines a new benchmark task for evaluating long-term memory in multi-session TOD. 

## Generation

### Multi-Session Dialogue Construction

Because existing TOD corpora typically feature single-session interactions lacking structured multi-session dependencies, we create three dialogue sessions for each task in the **SGD** dataset. Compared with single-session dialogues, this design more closely simulates how users revisit and refine the same task at different times and in different contexts. We chose three sessions rather than a higher number to strike a balance between capturing realistic user behavior and avoiding repetitive dialogue data, particularly given that SGD tasks involve fewer than ten task slots. As a result, three sessions offer sufficient coverage of task variations without overpopulating the dataset. The generation code can be found at `task_goal_oriented_dial_generation.py`.

### Confirmation-Type Response Annotation

In the final session of each task, we introduce confirmation-type annotations to mark utterances indicating the completion of long-term or recurring tasks. These annotations serve two primary functions:

* Guiding Memory Activation: They Highlight key dialogue points to trigger long-term memory activation, summaries, or confirmations.
* Supporting System Evaluation: They enable evaluation of the system’s ability to recognize and record cross-session information or long-term goals during dialogue strategy assessment. 

## Data

Personas are represented as a list of dialogue sessions, where each session contains a
list of utterances from either a user or an assistant and the annotation for the service, intent and confirmation. Since the MS-TOD dataset is focused on long-term memory in multiple sessions, there is only one service and one corresponding intent in a session.

Each persona is represented as a json object with the following fields:

*   **persona_id** - A unique identifier for a persona. 
*   **sessions** - A list of dialogue sessions.

Each session consists of the following fields:

*   **session_id** - A unique identifier for a session.
*   **reference_dialogue_id** - The `dialogue_id` of the reference dialogue from the SGD dataset used to generate the session.
*   **exist_confirmation** - An indicator of whether there exists a confirmation in the session.
*   **intent** - The name of the intent which is currently being fulfilled by the system in the session.
*   **service\*** - The name of the service present in the session.
*   **turns** - A list of annotated assistant or user utterances.
*   **confirmation state** - The dialogue state corresponding to the
    service when it is confirmed. If the value of `exist_confirmation` is `False`, it is an empty dictionary. Otherwise, it consists of the following fields:
    *   **slot_values** - A dictionary mapping slot name to a single value.
    *   **confirmation_utterance_id** - The identifier of the utterance that contains the confirmation in the session.

Each turn consists of the following fields:

*   **speaker** - The speaker for the turn. Possible values are "user" or
    "assistant".
*   **utterance** - A string containing the natural language utterance.

\*In the schema of the SGD dataset, `service_names` follow the form "\<domain name\>\_\<number\>" (e.g. Banks_2).
The number is used to disambiguate services from the same domain. To simplify, in the MS-TOD dataset, all the services from a same domain are combined to one service (e.g. both Banks_1 and Banks_2 are combined as Banks).

## Statistics
