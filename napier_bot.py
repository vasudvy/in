#!/usr/bin/env python3
"""
Napier Telegram Bot - An LLM-powered Telegram bot with web browsing capabilities.
Uses Gemini as the LLM and acts as an MCP client to communicate with MCP servers like Playwright.
"""

import os
import json
import asyncio
import logging
import subprocess
from typing import Dict, List, Optional, Union
import tempfile
import uuid
import base64
from datetime import datetime
from dotenv import load_dotenv

import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
load_dotenv()

# Load environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class MCPManager:
    """Manages MCP server connections"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.servers = {}
        self.active_servers = {}
        self.load_config()
        
    def load_config(self):
        """Load MCP server configurations from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.servers = config.get("mcpServers", {})
                logger.info(f"Loaded {len(self.servers)} MCP server configurations")
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            self.servers = {}
    
    async def connect_to_server(self, server_name: str, chat_id: str) -> Optional[ClientSession]:
        """Connect to an MCP server by name"""
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not found in config")
            return None
            
        server_config = self.servers[server_name]
        
        # If this chat already has an active server of this type, close it first
        if chat_id in self.active_servers and server_name in self.active_servers[chat_id]:
            await self.disconnect_server(chat_id, server_name)
        
        # Create new connection
        try:
            exit_stack = AsyncExitStack()
            
            # For Docker-based servers
            if server_config.get("command") == "docker":
                process = subprocess.Popen(
                    [server_config["command"]] + server_config["args"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    bufsize=0,
                )
                stdio_transport = await exit_stack.enter_async_context(
                    stdio_client(process.stdout, process.stdin)
                )
                
            # For direct commands
            elif "command" in server_config and "args" in server_config:
                server_params = StdioServerParameters(
                    command=server_config["command"],
                    args=server_config["args"],
                    env=None
                )
                stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
                
            # For URL-based servers
            elif "url" in server_config:
                # Implementation for HTTP/SSE transport would go here
                # This is a placeholder as we're focusing on stdio transport
                logger.error("URL-based servers not yet implemented")
                await exit_stack.aclose()
                return None
            else:
                logger.error(f"Invalid server configuration for {server_name}")
                await exit_stack.aclose()
                return None
                
            # Create client session
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(stdio, write))
            
            # Initialize session
            await session.initialize()
            
            # Store active server info
            if chat_id not in self.active_servers:
                self.active_servers[chat_id] = {}
                
            self.active_servers[chat_id][server_name] = {
                "session": session,
                "exit_stack": exit_stack
            }
            
            logger.info(f"Connected to MCP server {server_name} for chat {chat_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to connect to server {server_name}: {e}")
            return None
    
    async def disconnect_server(self, chat_id: str, server_name: str) -> bool:
        """Disconnect from an MCP server"""
        if chat_id not in self.active_servers or server_name not in self.active_servers[chat_id]:
            return False
            
        try:
            await self.active_servers[chat_id][server_name]["exit_stack"].aclose()
            del self.active_servers[chat_id][server_name]
            logger.info(f"Disconnected from MCP server {server_name} for chat {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from server {server_name}: {e}")
            return False
    
    async def disconnect_all(self, chat_id: str) -> bool:
        """Disconnect from all MCP servers for a chat"""
        if chat_id not in self.active_servers:
            return True
            
        success = True
        for server_name in list(self.active_servers[chat_id].keys()):
            if not await self.disconnect_server(chat_id, server_name):
                success = False
                
        if success:
            del self.active_servers[chat_id]
            
        return success
    
    def get_session(self, chat_id: str, server_name: str) -> Optional[ClientSession]:
        """Get an active MCP session"""
        if chat_id in self.active_servers and server_name in self.active_servers[chat_id]:
            return self.active_servers[chat_id][server_name]["session"]
        return None
    
    def get_available_servers(self) -> List[str]:
        """Get list of available MCP servers"""
        return list(self.servers.keys())

class NapierBot:
    """
    Napier Telegram Bot with LLM and MCP capabilities
    """
    
    def __init__(self):
        self.mcp_manager = MCPManager("napier_config.json")
        self.user_contexts = {}  # Store conversation contexts
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
            },
        )
        
    def get_user_context(self, chat_id: str) -> List:
        """Get or create user conversation context"""
        if chat_id not in self.user_contexts:
            self.user_contexts[chat_id] = []
        return self.user_contexts[chat_id]
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        chat_id = str(update.effective_chat.id)
        username = update.effective_user.username or update.effective_user.first_name
        
        # Reset user context
        self.user_contexts[chat_id] = []
        
        welcome_message = (
            f"👋 Hi *{username}*! I'm *Napier*, your AI assistant powered by LLM technology.\n\n"
            f"I can chat with you naturally and also help you browse the web using my web browsing capabilities.\n\n"
            f"Try asking me questions or use the /browse command to start web browsing!"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("Web Browsing", callback_data="cmd_browse"),
                InlineKeyboardButton("Help", callback_data="cmd_help"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_markdown(
            welcome_message, reply_markup=reply_markup
        )
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        help_text = (
            "*Napier Bot Commands*\n\n"
            "• /start - Initialize the bot\n"
            "• /help - Show this help message\n"
            "• /browse - Start web browsing session\n"
            "• /stop - Stop current web browsing session\n"
            "• /clear - Clear conversation history\n\n"
            "You can also simply chat with me naturally!"
        )
        
        await update.message.reply_markdown(help_text)
        
    async def browse_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /browse command"""
        chat_id = str(update.effective_chat.id)
        
        # Check if we have arguments (URL)
        if context.args and len(context.args) > 0:
            url = context.args[0]
            await self.start_browsing(update, chat_id, url)
        else:
            await update.message.reply_text(
                "Please enter a URL to browse. Example: /browse https://www.wikipedia.org"
            )
            
    async def start_browsing(self, update: Update, chat_id: str, url: str = None) -> None:
        """Start browsing session with Playwright MCP"""
        await update.message.reply_text("🔄 Connecting to web browser...")
        
        # Connect to the playwright MCP server
        session = await self.mcp_manager.connect_to_server("playwright", chat_id)
        
        if not session:
            await update.message.reply_text("❌ Failed to connect to web browser.")
            return
            
        # If URL is provided, navigate to it
        if url:
            try:
                await update.message.reply_text(f"🌐 Navigating to: {url}")
                await session.call_tool("browser_navigate", {"url": url})
                
                # Take snapshot
                snapshot_result = await session.call_tool("browser_snapshot", {})
                
                # Generate response about the page
                response = await self.generate_response(
                    chat_id,
                    f"I'm now browsing {url}. Here's what I can see on the page. Please describe it briefly and ask what I'd like to do next:",
                    snapshot_result.content
                )
                
                await update.message.reply_text(response)
                
            except Exception as e:
                logger.error(f"Error navigating to URL: {e}")
                await update.message.reply_text(f"❌ Error navigating to URL: {str(e)}")
                
        else:
            await update.message.reply_text(
                "🌐 Web browser is ready. You can ask me to browse to a specific URL."
            )
            
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stop command - stop current web browsing session"""
        chat_id = str(update.effective_chat.id)
        
        await update.message.reply_text("🔄 Stopping web browsing session...")
        success = await self.mcp_manager.disconnect_all(chat_id)
        
        if success:
            await update.message.reply_text("✅ Web browsing session stopped.")
        else:
            await update.message.reply_text("❌ Error stopping web browsing session.")
            
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command - clear conversation history"""
        chat_id = str(update.effective_chat.id)
        
        self.user_contexts[chat_id] = []
        await update.message.reply_text("🗑️ Conversation history cleared.")
        
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        command = query.data
        chat_id = str(update.effective_chat.id)
        
        if command == "cmd_browse":
            await query.edit_message_text(
                text="Please send a URL to browse or use /browse command."
            )
        elif command == "cmd_help":
            help_text = (
                "*Napier Bot Commands*\n\n"
                "• /start - Initialize the bot\n"
                "• /help - Show this help message\n"
                "• /browse - Start web browsing session\n"
                "• /stop - Stop current web browsing session\n"
                "• /clear - Clear conversation history\n\n"
                "You can also simply chat with me naturally!"
            )
            await query.edit_message_text(text=help_text, parse_mode="Markdown")
            
    async def save_image(self, image_data: str) -> str:
        """Save base64 image data to a file"""
        try:
            # Create temp directory if it doesn't exist
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate unique filename
            filename = f"{temp_dir}/image_{uuid.uuid4()}.png"
            
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
                
            # Decode and save image
            with open(filename, "wb") as f:
                f.write(base64.b64decode(image_data))
                
            return filename
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None
            
    async def generate_response(self, chat_id: str, message: str, context_info: str = None) -> str:
        """Generate response using Gemini LLM"""
        try:
            user_context = self.get_user_context(chat_id)
            
            # Create system message
            system_message = (
                "You are Napier, an intelligent assistant powered by Gemini LLM. "
                "You have web browsing capabilities through an MCP client that connects to a Playwright MCP server. "
                "When responding to users, highlight **key points** in your responses. "
                "Keep answers concise and helpful."
            )
            
            # Add system message to empty conversation
            if not user_context:
                user_context.append({"role": "system", "parts": [system_message]})
                
            # Add context info if provided (like web page content)
            if context_info:
                message = f"{message}\n\nHere's additional context:\n{context_info}"
                
            # Add user message
            user_context.append({"role": "user", "parts": [message]})
            
            # Generate response
            response = self.gemini_model.generate_content(user_context)
            
            # Extract response text
            response_text = response.text
            
            # Add assistant message to context
            user_context.append({"role": "model", "parts": [response_text]})
            
            # Limit context size
            if len(user_context) > 10:
                # Keep first (system) message and last 9 messages
                user_context = [user_context[0]] + user_context[-9:]
                
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error generating a response. Please try again. Error: {str(e)}"
            
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle normal messages"""
        chat_id = str(update.effective_chat.id)
        message_text = update.message.text
        
        # Check if message contains a URL and user might want to browse
        if message_text.startswith(("http://", "https://", "www.")):
            url = message_text
            if url.startswith("www."):
                url = "https://" + url
                
            await update.message.reply_text(f"I detected a URL. Browsing to {url}...")
            await self.start_browsing(update, chat_id, url)
            return
            
        # Process message for active browsing session or normal conversation
        playwright_session = self.mcp_manager.get_session(chat_id, "playwright")
        if playwright_session:
            # User has an active browsing session
            await update.message.reply_text("🔄 Processing your request...")
            
            try:
                # Take current snapshot
                snapshot_result = await playwright_session.call_tool("browser_snapshot", {})
                
                # Forward to LLM with special prompt to handle browsing actions
                browsing_prompt = (
                    f"The user's message is: '{message_text}'\n\n"
                    f"You have an active web browsing session. "
                    f"Analyze the current webpage snapshot and determine what action to take based on the user's request. "
                    f"Options include clicking elements, typing text, navigating to URLs, etc. "
                    f"If you need to use any of these Playwright MCP tools, format your response like this: "
                    f"TOOL: [tool_name] with ARGS: [arguments]"
                )
                
                response = await self.generate_response(chat_id, browsing_prompt, snapshot_result.content)
                
                # Parse response for potential tool calls
                if "TOOL:" in response and "ARGS:" in response:
                    # Extract tool name and arguments
                    tool_parts = response.split("TOOL:", 1)[1].split("ARGS:", 1)
                    tool_name = tool_parts[0].strip()
                    tool_args_str = tool_parts[1].strip()
                    
                    try:
                        # Parse arguments
                        tool_args = json.loads(tool_args_str)
                        
                        # Log tool call
                        logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                        
                        # Execute tool
                        await update.message.reply_text(f"Executing web action: {tool_name}")
                        result = await playwright_session.call_tool(tool_name, tool_args)
                        
                        # Handle special cases for certain tools
                        if tool_name == "browser_take_screenshot":
                            # Save and send screenshot
                            image_path = await self.save_image(result.content)
                            if image_path:
                                await update.message.reply_photo(photo=open(image_path, "rb"))
                                os.remove(image_path)  # Clean up
                                
                        # Take a new snapshot after action
                        new_snapshot = await playwright_session.call_tool("browser_snapshot", {})
                        
                        # Generate final response with updated snapshot
                        final_response = await self.generate_response(
                            chat_id,
                            "Describe what happened after the previous action and what the user can do next:",
                            new_snapshot.content
                        )
                        
                        await update.message.reply_text(final_response)
                        
                    except json.JSONDecodeError:
                        await update.message.reply_text(
                            "I had trouble parsing the required action. Could you please be more specific?"
                        )
                        
                else:
                    # No specific tool call found, return the generated response
                    await update.message.reply_text(response)
                    
            except Exception as e:
                logger.error(f"Error processing browsing request: {e}")
                await update.message.reply_text(f"❌ Error processing your request: {str(e)}")
                
        else:
            # Normal conversation
            await update.message.reply_chat_action("typing")
            response = await self.generate_response(chat_id, message_text)
            await update.message.reply_text(response, parse_mode="Markdown")
            
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        if update:
            await update.message.reply_text(
                "Sorry, something went wrong. Please try again later."
            )
            
    def run(self):
        """Run the bot"""
        if not TELEGRAM_TOKEN:
            logger.error("TELEGRAM_TOKEN environment variable not set")
            return
            
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY environment variable not set")
            return
            
        # Create application
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("browse", self.browse_command))
        application.add_handler(CommandHandler("stop", self.stop_command))
        application.add_handler(CommandHandler("clear", self.clear_command))
        
        # Add callback query handler
        application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Add message handler
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Add error handler
        application.add_error_handler(self.error_handler)
        
        # Run the bot
        application.run_polling()
        
if __name__ == "__main__":
    bot = NapierBot()
    bot.run()