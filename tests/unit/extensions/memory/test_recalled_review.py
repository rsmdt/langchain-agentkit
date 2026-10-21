from __future__ import annotations

import sys
import types
import uuid
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel

from langchain_agentkit import AgentKit, HITLExtension
from langchain_agentkit.extensions.memory import MemoryExtension, RecalledMemory

LIVE = ("memories",)
STAGING = ("staged_memories",)


class Memory(BaseModel):
    content: str


class RemoveDoc(BaseModel):
    json_doc_id: str


class _ExtractedMemory(tuple[str, Any]):
    @property
    def id(self) -> str:
        return str(self[0])

    @property
    def content(self) -> Any:
        return self[1]


def _manage_tool(*, namespace: Any, store: Any) -> StructuredTool:
    async def manage_memory(content: str) -> str:
        key = uuid.uuid4().hex
        await store.aput(tuple(namespace), key, {"content": content})
        return f"created {key}"

    return StructuredTool.from_function(
        coroutine=manage_memory, name="manage_memory", description="Record a memory."
    )


def _search_tool(*, namespace: Any, store: Any) -> StructuredTool:
    async def search_memory(query: str) -> str:
        return str([i.value for i in await store.asearch(tuple(namespace), query=query)])

    return StructuredTool.from_function(
        coroutine=search_memory, name="search_memory", description="Search memories."
    )


class _StoreManager:
    def __init__(self, model: Any, *, namespace: Any, store: Any) -> None:
        self.model = model
        self.namespace = namespace
        self.store = store


class _Executor:
    def __init__(self, reflector: Any, *, store: Any) -> None:
        self.reflector = reflector

    def submit(self, payload: Any, *, after_seconds: int) -> None:
        pass


class _ScriptedManager:
    def __init__(self, model: Any, kwargs: dict[str, Any], script: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        self.script = script
        self.inputs: list[Any] = []

    async def ainvoke(self, payload: dict[str, Any], config: Any = None) -> list[Any]:
        self.inputs.append(payload)
        return [_ExtractedMemory(pair) for pair in self.script(payload["existing"])]


def _namespace_template(namespace: Any) -> Any:
    parts = (namespace,) if isinstance(namespace, str) else tuple(namespace)

    def resolve(config: Any = None) -> tuple[str, ...]:
        configurable = (config or {}).get("configurable", {})
        return tuple(
            configurable[p[1:-1]] if p.startswith("{") and p.endswith("}") else p for p in parts
        )

    return resolve


@pytest.fixture
def fake_langmem(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("langmem")
    module.create_manage_memory_tool = _manage_tool  # type: ignore[attr-defined]
    module.create_search_memory_tool = _search_tool  # type: ignore[attr-defined]
    module.create_memory_store_manager = _StoreManager  # type: ignore[attr-defined]
    module.ReflectionExecutor = _Executor  # type: ignore[attr-defined]
    module.managers = []  # type: ignore[attr-defined]
    module.script = lambda existing: [(i, c) for i, _, c in existing]  # type: ignore[attr-defined]

    def create_memory_manager(model: Any, **kwargs: Any) -> _ScriptedManager:
        manager = _ScriptedManager(model, kwargs, module.script)  # type: ignore[attr-defined]
        module.managers.append(manager)  # type: ignore[attr-defined]
        return manager

    module.create_memory_manager = create_memory_manager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langmem", module)
    utils = types.ModuleType("langmem.utils")
    utils.NamespaceTemplate = _namespace_template  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langmem.utils", utils)
    return module


class _Runtime:
    def __init__(self, store: Any, config: Any = None) -> None:
        self.store = store
        self.config = config


class _StubModel(BaseChatModel):
    responses: list[AIMessage] = []
    idx: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> _StubModel:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        msg = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "stub"


def _snapshot(store: InMemoryStore, namespace: tuple[str, ...]) -> list[Any]:
    return [
        (i.namespace, i.key, i.value, i.created_at, i.updated_at)
        for i in store.search(namespace, limit=1000)
    ]


async def _seed(store: InMemoryStore, namespace: tuple[str, ...], **items: Any) -> None:
    for key, value in items.items():
        await store.aput(namespace, key, value)


@pytest.mark.asyncio
async def test_manage_tool_write_lands_only_after_hitl_approval(
    fake_langmem: types.ModuleType,
) -> None:
    store = InMemoryStore()
    call = {
        "name": "manage_memory",
        "args": {"content": "prefers tea"},
        "id": "c1",
        "type": "tool_call",
    }
    model = _StubModel(responses=[AIMessage("", tool_calls=[call]), AIMessage("done")])
    kit = AgentKit(
        extensions=[
            MemoryExtension(strategy=RecalledMemory(namespace=LIVE, store=store)),
            HITLExtension(interrupt_on={"manage_memory": True}, tools=[]),
        ],
        model=model,
    )

    async def handler(state: dict[str, Any], *, llm: Any, tools: Any, prompt: str) -> Any:
        bound = llm.bind_tools(tools)
        return {"messages": [await bound.ainvoke([SystemMessage(prompt), *state["messages"]])]}

    graph = kit.compile(handler).compile(checkpointer=InMemorySaver(), store=store)
    cfg = {"configurable": {"thread_id": "t"}}

    first = await graph.ainvoke({"messages": [HumanMessage("remember I like tea")]}, cfg)

    assert first.get("__interrupt__")
    assert await store.asearch(LIVE) == []

    await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), cfg)

    assert [i.value for i in await store.asearch(LIVE)] == [{"content": "prefers tea"}]


def test_staging_nested_under_live_namespace_is_rejected() -> None:
    with pytest.raises(ValueError, match="staging_namespace"):
        RecalledMemory(namespace=LIVE, staging_namespace=(*LIVE, "staged"))


@pytest.mark.asyncio
async def test_extraction_writes_to_staging_namespace(fake_langmem: types.ModuleType) -> None:
    store = InMemoryStore()
    strategy = RecalledMemory(
        namespace=LIVE,
        staging_namespace=STAGING,
        store=store,
        extract=True,
        extract_model="m",
    )

    await strategy.setup()

    assert strategy._executor.reflector.namespace == STAGING


@pytest.mark.asyncio
async def test_staged_memories_are_invisible_to_search_and_auto_recall(
    fake_langmem: types.ModuleType,
) -> None:
    store = InMemoryStore()
    await _seed(store, STAGING, s1={"kind": "Memory", "content": {"content": "staged fact"}})
    strategy = RecalledMemory(namespace=LIVE, staging_namespace=STAGING, store=store, auto_recall=5)

    await strategy.refresh(state={"messages": [HumanMessage("fact?")]}, runtime=_Runtime(store))
    searched = await strategy.tools[1].ainvoke({"query": "fact"})

    assert strategy.reminder() is None
    assert "staged fact" not in searched
    assert [i.key for i in await strategy.list_staged()] == ["s1"]


@pytest.mark.asyncio
async def test_list_staged_resolves_namespace_template(fake_langmem: types.ModuleType) -> None:
    store = InMemoryStore()
    await _seed(store, ("staged", "alice"), a={"content": "x"})
    await _seed(store, ("staged", "bob"), b={"content": "y"})
    strategy = RecalledMemory(
        namespace=("memories", "{user_id}"),
        staging_namespace=("staged", "{user_id}"),
        store=store,
    )

    staged = await strategy.list_staged(config={"configurable": {"user_id": "alice"}})

    assert [i.key for i in staged] == ["a"]


@pytest.mark.asyncio
async def test_approve_moves_staged_memory_into_live(fake_langmem: types.ModuleType) -> None:
    store = InMemoryStore()
    value = {"kind": "Memory", "content": {"content": "likes tea"}}
    await _seed(store, STAGING, s1=value, s2={"content": "other"})
    strategy = RecalledMemory(namespace=LIVE, staging_namespace=STAGING, store=store)

    await strategy.approve_staged("s1")

    assert await store.aget(STAGING, "s1") is None
    live = await store.aget(LIVE, "s1")
    assert live is not None
    assert live.value == value
    assert [i.key for i in await strategy.list_staged()] == ["s2"]


@pytest.mark.asyncio
async def test_reject_deletes_staged_memory(fake_langmem: types.ModuleType) -> None:
    store = InMemoryStore()
    await _seed(store, STAGING, s1={"content": "wrong"})
    strategy = RecalledMemory(namespace=LIVE, staging_namespace=STAGING, store=store)

    await strategy.reject_staged("s1")

    assert await store.aget(STAGING, "s1") is None
    assert await store.asearch(LIVE) == []


@pytest.mark.asyncio
async def test_approving_unknown_key_raises(fake_langmem: types.ModuleType) -> None:
    strategy = RecalledMemory(staging_namespace=STAGING, store=InMemoryStore())

    with pytest.raises(KeyError):
        await strategy.approve_staged("missing")


@pytest.mark.asyncio
async def test_review_requires_staging_namespace(fake_langmem: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="staging_namespace"):
        await RecalledMemory(store=InMemoryStore()).list_staged()


@pytest.mark.asyncio
async def test_consolidation_writes_only_to_target_and_leaves_source_untouched(
    fake_langmem: types.ModuleType,
) -> None:
    store = InMemoryStore()
    source = ("memories",)
    target = ("consolidated",)
    await _seed(
        store,
        source,
        keep={"content": "lives in Berlin"},
        dup_a={"content": "likes tea"},
        dup_b={"kind": "Memory", "content": {"content": "enjoys tea"}},
        old={"content": "prefers light mode"},
    )
    await _seed(store, ("other",), untouched={"content": "z"})

    def script(existing: list[Any]) -> list[Any]:
        passthrough = {i: c for i, _, c in existing}
        return [
            ("keep", passthrough["keep"]),
            ("dup_a", Memory(content="drinks tea daily")),
            ("dup_b", RemoveDoc(json_doc_id="dup_b")),
            ("old", RemoveDoc(json_doc_id="old")),
            ("new", Memory(content="prefers dark mode")),
        ]

    fake_langmem.script = script  # type: ignore[attr-defined]
    before = _snapshot(store, source)
    other_before = _snapshot(store, ("other",))
    strategy = RecalledMemory(namespace=source, store=store, extract_model="m")

    await strategy.consolidate(source=source, target=target)

    assert _snapshot(store, source) == before
    assert _snapshot(store, ("other",)) == other_before
    written = {i.key: i.value for i in await store.asearch(target)}
    assert written == {
        "keep": {"content": "lives in Berlin"},
        "dup_a": {"kind": "Memory", "content": {"content": "drinks tea daily"}},
        "new": {"kind": "Memory", "content": {"content": "prefers dark mode"}},
    }
    manager = fake_langmem.managers[0]  # type: ignore[attr-defined]
    assert manager.model == "m"
    assert sorted(manager.inputs[0]["existing"]) == [
        ("dup_a", "Memory", {"content": "likes tea"}),
        ("dup_b", "Memory", {"content": "enjoys tea"}),
        ("keep", "Memory", {"content": "lives in Berlin"}),
        ("old", "Memory", {"content": "prefers light mode"}),
    ]


@pytest.mark.asyncio
async def test_consolidation_reads_only_the_exact_source_namespace(
    fake_langmem: types.ModuleType,
) -> None:
    store = InMemoryStore()
    await _seed(store, ("memories",), a={"content": "a"})
    await _seed(store, ("memories", "nested"), b={"content": "b"})
    strategy = RecalledMemory(store=store, extract_model="m")

    await strategy.consolidate(source=("memories",), target=("consolidated",))

    assert [i.key for i in await store.asearch(("consolidated",))] == ["a"]


@pytest.mark.asyncio
async def test_consolidation_accepts_an_explicit_model(fake_langmem: types.ModuleType) -> None:
    store = InMemoryStore()
    await _seed(store, ("memories",), a={"content": "a"})
    strategy = RecalledMemory(store=store)

    await strategy.consolidate(source=("memories",), target=("out",), model="other")

    assert fake_langmem.managers[0].model == "other"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_consolidation_requires_a_model(fake_langmem: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="model"):
        await RecalledMemory(store=InMemoryStore()).consolidate(
            source=("memories",), target=("out",)
        )


@pytest.mark.asyncio
async def test_consolidation_refuses_target_equal_to_source(
    fake_langmem: types.ModuleType,
) -> None:
    with pytest.raises(ValueError, match="target"):
        await RecalledMemory(store=InMemoryStore(), extract_model="m").consolidate(
            source=("memories",), target=("memories",)
        )


@pytest.mark.asyncio
async def test_consolidation_refuses_non_empty_target(fake_langmem: types.ModuleType) -> None:
    store = InMemoryStore()
    await _seed(store, ("memories",), a={"content": "a"})
    await _seed(store, ("out",), stale={"content": "old run"})
    strategy = RecalledMemory(store=store, extract_model="m")

    with pytest.raises(ValueError, match="not empty"):
        await strategy.consolidate(source=("memories",), target=("out",))

    assert fake_langmem.managers == []  # type: ignore[attr-defined]
