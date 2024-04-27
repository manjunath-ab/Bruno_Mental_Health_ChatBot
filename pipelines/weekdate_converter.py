from datetime import datetime, timedelta

def convert_to_iso8601(weekday_time):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    current_date = datetime.now()

    # Find the index of the given weekday
    given_weekday = weekday_time[0]
    weekday_index = weekdays.index(given_weekday)

    # Calculate the difference in days between the given weekday and the current weekday
    days_difference = (weekday_index - current_date.weekday()) % 7

    # Adjust the current date by the calculated difference
    target_date = current_date + timedelta(days=days_difference)

    # Extract time from the input
    given_time = datetime.strptime(weekday_time[1], '%I:%M %p').time()

    # Combine date and time
    combined_datetime = datetime.combine(target_date, given_time)

    # Convert to ISO 8601 format with timezone offset
    #iso8601_formatted = combined_datetime.strftime('%Y-%m-%dT%H:%M:%S%z')

    return combined_datetime

# Test the function
#weekday_time = ('Friday', '09:00 AM')
#print(convert_to_iso8601(weekday_time))
