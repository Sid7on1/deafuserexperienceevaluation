import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkshopStatus(Enum):
    """Enum for workshop status"""
    PENDING = 1
    IN_PROGRESS = 2
    COMPLETED = 3

@dataclass
class Participant:
    """Data class for participant information"""
    id: int
    name: str
    email: str
    phone_number: str

class WorkshopException(Exception):
    """Custom exception for workshop-related errors"""
    pass

class UserWorkshop:
    """Main class for user workshop methodology"""
    def __init__(self, workshop_id: int, workshop_name: str, workshop_date: datetime):
        """
        Initialize the UserWorkshop class

        Args:
        - workshop_id (int): Unique identifier for the workshop
        - workshop_name (str): Name of the workshop
        - workshop_date (datetime): Date of the workshop
        """
        self.workshop_id = workshop_id
        self.workshop_name = workshop_name
        self.workshop_date = workshop_date
        self.participants: List[Participant] = []
        self.status = WorkshopStatus.PENDING

    def workshop_facilitation(self, facilitator_name: str, facilitator_email: str):
        """
        Facilitate the workshop

        Args:
        - facilitator_name (str): Name of the facilitator
        - facilitator_email (str): Email of the facilitator

        Raises:
        - WorkshopException: If the workshop is not in the pending state
        """
        if self.status != WorkshopStatus.PENDING:
            raise WorkshopException("Workshop is not in the pending state")
        logging.info(f"Workshop {self.workshop_name} is being facilitated by {facilitator_name}")
        self.status = WorkshopStatus.IN_PROGRESS

    def participant_recruitment(self, participant: Participant):
        """
        Recruit a participant for the workshop

        Args:
        - participant (Participant): Participant information

        Raises:
        - WorkshopException: If the workshop is not in the pending state
        """
        if self.status != WorkshopStatus.PENDING:
            raise WorkshopException("Workshop is not in the pending state")
        self.participants.append(participant)
        logging.info(f"Participant {participant.name} has been recruited for workshop {self.workshop_name}")

    def data_collection(self):
        """
        Collect data from the workshop participants

        Returns:
        - data (Dict): Collected data from the participants

        Raises:
        - WorkshopException: If the workshop is not in the in_progress state
        """
        if self.status != WorkshopStatus.IN_PROGRESS:
            raise WorkshopException("Workshop is not in the in_progress state")
        data = {}
        for participant in self.participants:
            # Simulate data collection from participants
            participant_data = {
                "participant_id": participant.id,
                "participant_name": participant.name,
                "participant_email": participant.email,
                "participant_phone_number": participant.phone_number
            }
            data[participant.id] = participant_data
        logging.info(f"Data has been collected from {len(self.participants)} participants")
        self.status = WorkshopStatus.COMPLETED
        return data

    def get_workshop_status(self) -> WorkshopStatus:
        """
        Get the current status of the workshop

        Returns:
        - status (WorkshopStatus): Current status of the workshop
        """
        return self.status

    def get_participant_list(self) -> List[Participant]:
        """
        Get the list of participants for the workshop

        Returns:
        - participants (List[Participant]): List of participants
        """
        return self.participants

def main():
    # Create a new workshop
    workshop = UserWorkshop(1, "Deaf User Experience Evaluation", datetime(2024, 9, 16))

    # Recruit participants
    participant1 = Participant(1, "John Doe", "john.doe@example.com", "123-456-7890")
    participant2 = Participant(2, "Jane Doe", "jane.doe@example.com", "987-654-3210")
    workshop.participant_recruitment(participant1)
    workshop.participant_recruitment(participant2)

    # Facilitate the workshop
    workshop.workshop_facilitation("Facilitator Name", "facilitator@example.com")

    # Collect data from participants
    data = workshop.data_collection()
    print(data)

    # Get workshop status and participant list
    print(workshop.get_workshop_status())
    print(workshop.get_participant_list())

if __name__ == "__main__":
    main()