"""
Command-line interface for the oscilloscope MCP server.
"""

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
import structlog

from .server import OscilloscopeMCPServer

# Initialize rich console
console = Console()

# Configure logging with rich
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Create Typer app
app = typer.Typer(
    name="oscilloscope-mcp",
    help="Professional oscilloscope and function generator MCP server",
    add_completion=False,
)

@app.command()
def serve(
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind to"),
    hardware_interface: str = typer.Option("simulation", "--interface", "-i", help="Hardware interface type"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level"),
):
    """Start the MCP server."""
    console.print("[bold green]Starting Oscilloscope MCP Server[/bold green]")
    console.print(f"Interface: {hardware_interface}")
    console.print(f"Log level: {log_level}")
    
    # Set log level
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    
    # Run the server
    asyncio.run(run_server(hardware_interface))

@app.command()
def test(
    hardware_interface: str = typer.Option("simulation", "--interface", "-i", help="Hardware interface type"),
):
    """Run self-test on the hardware interface."""
    console.print("[bold blue]Running Hardware Self-Test[/bold blue]")
    
    async def run_test():
        server = OscilloscopeMCPServer()
        try:
            # Initialize with test config
            test_config = {
                "hardware_interface": hardware_interface,
                "log_level": "INFO",
                "sample_rate": 1e6,  # 1 MS/s for testing
                "channels": 4,
                "buffer_size": 1000,
                "timeout": 5.0,
            }
            
            await server.hardware.initialize(test_config)
            
            # Run self-test
            console.print("Running self-test...")
            test_results = await server.hardware.perform_self_test()
            
            # Display results
            if test_results["overall_status"] == "pass":
                console.print("[bold green]✓ Self-test PASSED[/bold green]")
            else:
                console.print("[bold red]✗ Self-test FAILED[/bold red]")
            
            console.print("\nTest Results:")
            for test_name, result in test_results["tests"].items():
                if result == "pass":
                    console.print(f"  [green]✓[/green] {test_name}")
                else:
                    console.print(f"  [red]✗[/red] {test_name}: {result}")
            
        except Exception as e:
            console.print(f"[bold red]Test failed with error: {e}[/bold red]")
        finally:
            await server.hardware.cleanup()
    
    asyncio.run(run_test())

@app.command()
def version():
    """Show version information."""
    from . import __version__, __description__
    console.print(f"Oscilloscope MCP Server v{__version__}")
    console.print(__description__)

async def run_server(hardware_interface: str):
    """Run the MCP server."""
    server = OscilloscopeMCPServer()
    
    try:
        console.print("Initializing server...")
        await server.start()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Received interrupt signal[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Server error: {e}[/bold red]")
        logger.error("Server error", error=str(e))
        sys.exit(1)
    finally:
        console.print("Shutting down server...")
        await server.stop()
        console.print("[green]Server stopped[/green]")

def main():
    """Main entry point."""
    app()

if __name__ == "__main__":
    main()