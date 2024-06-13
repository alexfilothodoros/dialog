import os
import pytest

from dialog_lib.db.models import ChatMessages, Chat
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from dialog.llm.agents import lcel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from dialog_lib.embeddings.retrievers import DialogRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Dialog API is healthy"}

def test_create_chat_session(client, mocker):
    response = client.post("/session")
    assert response.status_code == 200
    assert "chat_id" in response.json()

def test_post_message_no_session_id(client, chat_session, mocker, llm_mock, dbsession):
    session_id = chat_session["chat_id"]
    response = client.post(f"/chat/{session_id}", json={"message": "Hello"})
    assert llm_mock.called
    assert response.status_code == 200
    assert "message" in response.json()


def test_ask_question_no_session_id(client, mocker, llm_mock, dbsession):
    response = client.post("/ask", json={"message": "Hello"})
    assert response.status_code == 200
    assert "message" in response.json()
    assert llm_mock.called
    assert dbsession.query(Chat).count() == 0

def test_get_chat_content(client, chat_session, dbsession):
    chat = ChatMessages(session_id=chat_session["chat_id"], message="Hello")
    dbsession.add(chat)
    dbsession.flush()
    response = client.get(f"/chat/{chat_session['chat_id']}")
    assert response.status_code == 200
    assert "message" in response.json()
    assert dbsession.query(ChatMessages).count() == 1

def test_get_all_sessions(client, chat_session):
    response = client.get("/sessions")
    assert response.status_code == 200
    assert "sessions" in response.json()
    assert len(response.json()["sessions"]) == 1

def test_invalid_database_connection(client, mocker):
    mocker.patch("dialog.routers.dialog.engine.connect", side_effect=Exception)
    with pytest.raises(Exception):
        response = client.get("/health")
        assert response.status_code == 500
        assert response.json() == {"message": "Failed to execute simple SQL"}

# test openai router

def test_customized_openai_models_response(client):
    response = client.get("/openai/models")
    assert response.status_code == 200
    for i in ["id", "object", "created", "owned_by"]:
        assert i in response.json()[0]


def test_customized_openai_chat_completion_response_stream_false(client, llm_mock_openai_router):
    os.environ["LLM_CLASS"] = "dialog.llm.agents.default.DialogLLM"
    response = client.post("/openai/chat/completions", json={
        "model": "talkd-ai",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "stream": False
    })
    assert response.status_code == 200
    for i in ["choices", "created", "id", "model", "object", "usage"]:
        assert i in response.json()
    assert llm_mock_openai_router.called
    assert response.json()["choices"][0]["message"]["role"] == "assistant"

def test_multiple_models_are_available_on_model_listing_for_webui(client_with_settings_override):
    response = client_with_settings_override.get("/openai/models")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["id"] == "talkd-ai"
    assert response.json()[1]["id"] == "blob_model"

def test_multiple_models_are_usable_on_chat_completion(client_with_settings_override, mocker):
    process_user_message_mock = mocker.patch(
        "dialog.routers.openai.process_user_message",
        return_value={"text": "Hello"}
    )
    response = client_with_settings_override.post(
            "/openai/chat/completions",
            json={
            "model": "blob_model",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ],
            "stream": False
        }
    )
    assert process_user_message_mock.call_args[1]['message'] == "Hello"
    assert process_user_message_mock.call_args[1]['model_class_path'] == "dialog.llm.agents.default.DialogLLM"
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["role"] == "assistant"
    assert response.json()["choices"][0]["message"]["content"] == "Hello"

def test_unknown_model_return_error_on_chat_completion(client_with_settings_override, mocker):
    process_user_message_mock = mocker.patch(
        "dialog.routers.openai.process_user_message",
        return_value={"text": "Hello"}
    )
    response = client_with_settings_override.post(
            "/openai/chat/completions",
            json={
            "model": "unknown-model",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ],
            "stream": False
        }
    )
    assert response.status_code == 404


# Test for get_memory_instance
@patch('dialog.llm.agents.lcel.get_session')
@patch('dialog.llm.agents.lcel.generate_memory_instance')
def test_get_memory_instance(mock_generate_memory_instance, mock_get_session):
    mock_get_session.return_value = iter([MagicMock()])
    lcel.get_memory_instance('test_session')
    mock_generate_memory_instance.assert_called_once()
    assert mock_generate_memory_instance.call_args[1]['session_id'] == 'test_session'
    

# Test for retriever
def test_retriever():
    assert isinstance(lcel.retriever, DialogRetriever)
    assert lcel.retriever.session is not None
    assert lcel.retriever.embedding_llm is not None


# Test for format_docs
def test_format_docs():
    docs = [Document(page_content='doc1'), Document(page_content='doc2')]
    result = lcel.format_docs(docs)
    assert result == 'doc1\n\ndoc2'
    assert isinstance(result, str)


# Test for runnable
def test_runnable():
    assert isinstance(lcel.runnable, RunnableWithMessageHistory)
    assert lcel.runnable.input_messages_key == 'input'
    assert lcel.runnable.history_messages_key == 'chat_history'


@pytest.fixture
def psql_memory():
    import pytest
    from unittest.mock import Mock
    from dialog.learn.helpers import stopwords

    psql_memory = Mock()
    psql_memory.messages = [
        Mock(content="This is message 1"),
        Mock(content="This is message 2"),
        Mock(content="This is message 3"),
    ]
    return psql_memory


async def test_gen():
    from src.dialog.routers.openai import ask_question_to_llm
    from unittest.mock import MagicMock
    from dialog.learn.helpers import stopwords
    from dialog.learn.idf import categorize_conversation_history

    _g = ask_question_to_llm(message="Hello")
    expected_output = [
        'data: {"id": "talkdai-<random_uuid>", "choices": [{"index": 0, "delta": {"content": "word "}}]}\n\n',
        'data: {"id": "talkdai-<random_uuid>", "choices": [{"index": 0, "delta": {"content": "+END "}}]}\n\n'
    ]
    output = list(_g.gen())
    assert output == expected_output
